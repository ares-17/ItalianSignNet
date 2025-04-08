from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import stack, tensor, no_grad
import os
from dotenv import load_dotenv
from datasets import ClassLabel
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

REPO_MODEL = "bazyl/gtsrb-model"

def get_latest_dataset_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(script_dir, 'src', 'dataset', 'artifacts')
    all_subdirs = [
        d for d in os.listdir(artifacts_dir) 
        if os.path.isdir(os.path.join(artifacts_dir, d))
    ]
    
    if not all_subdirs:
        raise ValueError("Nessuna cartella dataset trovata nella directory artifacts")
    
    return os.path.join(artifacts_dir, sorted(all_subdirs)[-1])

def preprocess(processor, examples):
    inputs = processor(images=examples["image"], return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"].squeeze(0),
        "labels": examples["label"]
    }

def collate_fn(batch):
    return {
        "pixel_values": stack([tensor(x["pixel_values"]) for x in batch]),
        "labels": tensor([x["labels"] for x in batch])
    }

def evaluate(model, test_dataloader):
    all_preds = []
    all_labels = []

    with no_grad():
        for batch in test_dataloader:
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            logits = outputs.logits
            preds = logits.argmax(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            
    return all_preds, all_labels

def mlflow_tracking(all_labels, all_preds, datasets_folder):
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    mlflow.set_tracking_uri('http://localhost:5000')

    # Logging su MLflow
    with mlflow.start_run() as run:
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        
        # Log della confusion matrix come artefatto
        np.savetxt("confusion_matrix.csv", conf_matrix, delimiter=",")
        mlflow.log_artifact("confusion_matrix.csv")
        
        # Log dei parametri del modello
        mlflow.log_param("model_name", "bazyl/gtsrb-model")
        mlflow.log_param("dataset_version", os.path.basename(datasets_folder))

    print("Metriche loggate con successo:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

def main():
    load_dotenv()
    
    datasets_folder = get_latest_dataset_folder()
    dataset = load_dataset("imagefolder", data_dir=datasets_folder)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=43))

    processor = AutoImageProcessor.from_pretrained(REPO_MODEL)
    model = AutoModelForImageClassification.from_pretrained(REPO_MODEL)
    
    dataset = dataset.map((lambda x: preprocess(processor, x)), batched=False, remove_columns=["image"])
    test_dataloader = DataLoader(dataset["test"], batch_size=4, collate_fn=collate_fn)
    model.eval()

    all_preds, all_labels = evaluate(model, test_dataloader)
    mlflow_tracking(all_labels, all_preds, datasets_folder)
    
if __name__ == "__main__":
    main()