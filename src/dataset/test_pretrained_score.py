import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import mlflow
import torch
from datasets import (
    DatasetDict,
    load_dataset,
    ClassLabel,
    Features,
    Value,
    Image,  
)
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import Tensor, no_grad, stack, tensor
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    ImageProcessingMixin,
    PreTrainedModel,
)
import seaborn as sns
import os
import pandas as pd
from datasets import DatasetDict, load_dataset, ClassLabel
from typing import Dict, Any
import numpy as np
import logging
import datetime

REPO_MODEL: str = "bazyl/gtsrb-model"

# use or create log's dir
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# logger definition
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(
        os.path.join(logs_dir, f"test_pretrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}_eps_{REPO_MODEL.replace('/', '_')}.log")
    )]
)
LOGGER = logging.getLogger(__name__)

def get_latest_dataset_folder() -> Tuple[str, str]:
    """
    Restituisce il percorso della cartella dataset più recente all'interno della directory degli artifacts.
    """
    BASE_DIR = Path(os.getenv("BASE_DIR", ""))
    artifacts_dir = BASE_DIR / "src" / "dataset" / "artifacts"

    all_subdirs = [
        d for d in os.listdir(artifacts_dir) if (artifacts_dir / d).is_dir()
    ]
    if not all_subdirs:
        raise ValueError("Nessuna cartella dataset trovata nella directory artifacts")

    newest = sorted(all_subdirs)[-1]
    LOGGER.info(f"Dataset selezionato: {newest}")
    return str(artifacts_dir / newest), newest

def preprocess(
    processor: ImageProcessingMixin, examples: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Preprocessa un campione del dataset estraendo i pixel e le etichette in un formato adatto al modello.
    """
    inputs = processor(images=examples["image"], return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"].squeeze(),
        "labels": examples["label"],
    }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Funzione di collazione per il DataLoader.
    """
    return {
        "pixel_values": stack([tensor(x["pixel_values"]) for x in batch]),
        "labels": tensor([x["labels"] for x in batch]),
    }

def evaluate(
    model: PreTrainedModel, test_dataloader: DataLoader
) -> Tuple[List[int], List[int], List[float], List[int]]:
    """
    Valuta il modello sul test set.
    """
    all_preds: List[int] = []
    all_labels: List[int] = []

    class_correct: List[int] = [0] * 43
    class_total: List[int] = [0] * 43

    with no_grad():
        for batch in test_dataloader:
            outputs = model(pixel_values=batch["pixel_values"])
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            _, preds = torch.max(probs, dim=1)

            for i in range(len(preds)):
                label = batch["labels"][i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    present_classes = sorted(list(set(all_labels)))
    LOGGER.info("\nAccuracy per classe:")
    for cls in present_classes:
        accuracy = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        LOGGER.info(f"Classe {cls}: {accuracy:.4f} ({class_correct[cls]}/{class_total[cls]})")

    return all_preds, all_labels, present_classes

def plot_confusion_matrix(all_labels, all_preds, present_classes):
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=present_classes)
    
    # Create a custom annotation array that leaves cells blank if the value is 0
    annot = np.where(conf_matrix == 0, "", conf_matrix)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=annot, fmt="", cmap="Blues", ax=ax, cbar=False)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.xaxis.set_ticklabels(present_classes)
    ax.yaxis.set_ticklabels(present_classes)
    fig.tight_layout()
    
    return fig

def mlflow_tracking(
    all_labels: List[int],
    all_preds: List[int],
    datasets_folder: str,
    folder_name: str,
    present_classes: List[int],
) -> None:
    """
    Registra metriche e artefatti su MLflow, inclusi report dettagliati.
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=present_classes, average="weighted", zero_division=0
    )

    mlflow.set_tracking_uri("http://localhost:5000")

    with mlflow.start_run(run_name=f"{REPO_MODEL}_{folder_name}"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_figure(plot_confusion_matrix(all_labels, all_preds, present_classes), "confusion_matrix.png")
        mlflow.log_param("model_name", REPO_MODEL)
        mlflow.log_param("dataset_version", os.path.basename(datasets_folder))

    LOGGER.info("Metriche loggate con successo:")
    LOGGER.info(f"Accuracy: {accuracy:.4f}")
    LOGGER.info(f"Precision: {precision:.4f}")
    LOGGER.info(f"Recall: {recall:.4f}")
    LOGGER.info(f"F1-Score: {f1:.4f}")

def validate_dataset_structure(dataset: DatasetDict) -> None:
    """
    Verifica che il dataset abbia la struttura corretta.
    """
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Il dataset deve contenere split 'train' e 'test'")

    train_labels = set(example["label"] for example in dataset["train"])
    test_labels = set(example["label"] for example in dataset["test"])

    LOGGER.info(f"Numero di classi nel train set: {len(train_labels)}")
    LOGGER.info(f"Numero di classi nel test set: {len(test_labels)}")

    train_only = train_labels - test_labels
    test_only = test_labels - train_labels

    if train_only:
        LOGGER.info(f"Attenzione: Classi presenti solo nel train set: {sorted(train_only)}")
    if test_only:
        LOGGER.info(f"Attenzione: Classi presenti solo nel test set: {sorted(test_only)}")
 
def load_dataset_with_features(data_dir: str, metadata_path: str) -> DatasetDict:
    metadata = pd.read_parquet(metadata_path)
    filename_to_label = dict(zip(metadata['filename'], metadata['feature_index'].astype(int)))

    features = Features({
        'image': Image(),
        'label': Value('int64')
    })

    dataset = load_dataset(
        "imagefolder",
        data_dir=data_dir,
        drop_labels=True,
        features=features
    )

    def add_right_labels(example: Dict[str, Any]) -> Dict[str, Any]:
        filename = os.path.basename(os.path.basename(example['image'].filename))
        example['label'] = filename_to_label[filename]
        return example

    for split in dataset:
        dataset[split] = dataset[split].map(add_right_labels)

    return dataset.cast_column("label", ClassLabel(num_classes=43))

def main() -> None:    
    load_dotenv()
    datasets_folder, folder_name = get_latest_dataset_folder()
    metadata_path = os.path.join(datasets_folder, "metadata.parquet")

    dataset = load_dataset_with_features(
        data_dir=datasets_folder,
        metadata_path=metadata_path
    )

    processor = AutoImageProcessor.from_pretrained(REPO_MODEL)
    model: PreTrainedModel = AutoModelForImageClassification.from_pretrained(REPO_MODEL)
    model.eval()

    LOGGER.info("Specifiche di normalizzazione del modello:")
    LOGGER.info(f"Normalizzazione: {processor.do_normalize}")
    LOGGER.info(f"Valori medi: {processor.image_mean}")
    LOGGER.info(f"Valori std: {processor.image_std}")
    LOGGER.info(f"Dimensione immagine: {processor.size}")

    validate_dataset_structure(dataset)

    dataset = dataset.map(lambda x: preprocess(processor, x), batched=True, remove_columns=["image"])
    test_dataloader = DataLoader(dataset["test"], batch_size=4, collate_fn=collate_fn)

    all_preds, all_labels, present_classes = evaluate(model, test_dataloader)
    mlflow_tracking(all_labels, all_preds, datasets_folder, folder_name, present_classes)

if __name__ == "__main__":
    main()
