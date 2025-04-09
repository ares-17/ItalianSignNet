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
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_MODEL = "bazyl/gtsrb-model"

def get_latest_dataset_folder():
    """
    Restituisce il percorso della cartella dataset più recente all'interno della directory degli artifacts.
    
    Cerca all'interno della cartella 'src/dataset/artifacts' quella subdirectory che, ordinata alfabeticamente, risulta essere l'ultima.
    Se non viene trovata nessuna cartella, solleva un'eccezione.
    
    Returns:
        str: Il percorso assoluto della cartella dataset più recente.
    
    Raises:
        ValueError: Se non vengono trovate sottocartelle all'interno di 'artifacts'.
    """
    BASE_DIR = Path(os.getenv("BASE_DIR"))
    artifacts_dir = os.path.join(BASE_DIR, 'src', 'dataset', 'artifacts')
    all_subdirs = [
        d for d in os.listdir(artifacts_dir) 
        if os.path.isdir(os.path.join(artifacts_dir, d))
    ]
    
    if not all_subdirs:
        raise ValueError("Nessuna cartella dataset trovata nella directory artifacts")
    
    newest = sorted(all_subdirs)[-1]
    return os.path.join(artifacts_dir, newest), newest

def preprocess(processor, examples):
    """
    Preprocessa un esempio del dataset estraendo i pixel e le etichette in un formato adatto al modello.
    
    Args:
        processor: L'istanza del processor (AutoImageProcessor) usato per gestire il preprocessing delle immagini.
        examples (dict): Un dizionario che contiene almeno le chiavi "image" e "label".
    
    Returns:
        dict: Un dizionario contenente:
            - "pixel_values": il tensore delle immagini processate
            - "labels": l'etichetta associata all'immagine.
    
    Nota:
        L'uso di `return_tensors="pt"` converte l'immagine in un tensore PyTorch.
    """
    inputs = processor(images=examples["image"], return_tensors="pt")
    # Debug: stampa la forma del tensore
    print(f"Forma del tensore di input: {inputs['pixel_values'].shape}")
    return {
        "pixel_values": inputs["pixel_values"].squeeze(),
        "labels": examples["label"]
    }

def collate_fn(batch):
    """
    Funzione di collazione per il DataLoader, combinando più esempi in un singolo batch.
    
    Args:
        batch (list): Una lista di dizionari, ciascuno contenente "pixel_values" e "labels".
    
    Returns:
        dict: Un dizionario con:
            - "pixel_values": un tensore creato impilando i tensori di ciascun esempio.
            - "labels": un tensore contenente le etichette, ricostruito da ciascun esempio.
    
    Nota:
        L'uso di stack e tensor assicura che il batch sia formato correttamente per il modello.
    """
    return {
        "pixel_values": stack([tensor(x["pixel_values"]) for x in batch]),
        "labels": tensor([x["labels"] for x in batch])
    }

def evaluate(model, test_dataloader):
    """
    Valuta il modello sul test set, accumulando le predizioni e le etichette.
    
    Args:
        model: Il modello di classificazione, ad es. un'istanza di AutoModelForImageClassification.
        test_dataloader: Il DataLoader che fornisce i batch del test set.
    
    Returns:
        tuple: Una tupla contenente due liste:
            - all_preds: Predizioni fatte dal modello su tutte le immagini.
            - all_labels: Etichette reali per le immagini del test set.
    
    Nota:
        - La funzione disabilita il calcolo del gradiente con `no_grad()` per migliorare le performance durante l'inferenza.
        - Le predizioni vengono ottenute con `argmax` sui logits per ottenere la classe con massima probabilità.
    """
    all_preds = []
    all_labels = []
    confidence_scores = []
    class_correct = [0] * 43  # Per 43 classi del GTSRB
    class_total = [0] * 43
    
    with no_grad():
        for batch in test_dataloader:
            outputs = model(pixel_values=batch["pixel_values"])
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, preds = torch.max(probs, dim=1)
            
            # Calcola statistiche per classe
            for i in range(len(preds)):
                label = batch["labels"][i].item()
                pred = preds[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            confidence_scores.extend(confidence.cpu().numpy())
    
    # Calcola accuratezza per classe
    for i in range(43):
        accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"Classe {i}: {accuracy:.4f} ({class_correct[i]}/{class_total[i]})")
    
    # Analizza predizioni con bassa confidenza
    low_confidence_indices = [i for i, conf in enumerate(confidence_scores) if conf < 0.5]
    print(f"Predizioni con bassa confidenza: {len(low_confidence_indices)}")
    
    return all_preds, all_labels, confidence_scores

def mlflow_tracking(all_labels, all_preds, datasets_folder, folder_name):
    """
    Calcola le metriche di valutazione e registra le metriche e gli artefatti (come la matrice di confusione)
    su MLflow.
    
    Args:
        all_labels (list): Lista delle etichette reali.
        all_preds (list): Lista delle predizioni fatte dal modello.
        datasets_folder (str): Il percorso della cartella del dataset per tracciare la versione del dataset.
        folder_name (str): Nome della cartella che contiene il dataset
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    mlflow.set_tracking_uri('http://localhost:5000')

    # Logging su MLflow
    with mlflow.start_run(run_name=f"{REPO_MODEL}_{folder_name}") as run:
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

def visualize_preprocessing(processor, dataset_path):
    # Carica alcune immagini esempio
    dataset = load_dataset("imagefolder", data_dir=dataset_path)
    
    # Prendi 3 immagini di esempio
    samples = [dataset["train"][i]["image"] for i in range(3)]
    
    # Visualizza le immagini originali
    plt.figure(figsize=(15, 10))
    for i, img in enumerate(samples):
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"Originale {i}")
    
    # Processa le immagini
    processed = processor(images=samples, return_tensors="pt")
    
    # Visualizza le immagini processate (denormalizzate per la visualizzazione)
    for i, img_tensor in enumerate(processed["pixel_values"]):
        # Denormalizza
        img_array = img_tensor.permute(1, 2, 0).numpy()
        img_array = img_array * np.array(processor.image_std) + np.array(processor.image_mean)
        img_array = np.clip(img_array, 0, 1)
        
        plt.subplot(2, 3, i+4)
        plt.imshow(img_array)
        plt.title(f"Processata {i}")
    
    plt.tight_layout()
    plt.savefig("preprocessing_comparison.png")
    plt.close()
    
    print("Confronto salvato in 'preprocessing_comparison.png'")

def validate_dataset_structure(dataset):
    """Verifica che il dataset abbia la struttura corretta"""
    # Verifica la presenza di train e test
    if "train" not in dataset or "test" not in dataset:
        raise ValueError("Il dataset deve contenere split 'train' e 'test'")
    
    # Verifica il numero di classi
    train_labels = set([example["label"] for example in dataset["train"]])
    test_labels = set([example["label"] for example in dataset["test"]])
    
    print(f"Numero di classi nel train set: {len(train_labels)}")
    print(f"Numero di classi nel test set: {len(test_labels)}")
    
    # Verifica se ci sono classi che non sono presenti in entrambi gli split
    train_only = train_labels - test_labels
    test_only = test_labels - train_labels
    
    if train_only:
        print(f"Attenzione: Classi presenti solo nel train set: {sorted(train_only)}")
    
    if test_only:
        print(f"Attenzione: Classi presenti solo nel test set: {sorted(test_only)}")
    
    return True

def main():
    load_dotenv()
    
    datasets_folder, folder_name = get_latest_dataset_folder()
    dataset = load_dataset("imagefolder", data_dir=datasets_folder)
    dataset = dataset.cast_column("label", ClassLabel(num_classes=43))

    processor = AutoImageProcessor.from_pretrained(REPO_MODEL)
    model = AutoModelForImageClassification.from_pretrained(REPO_MODEL)
    
    processor = AutoImageProcessor.from_pretrained("bazyl/gtsrb-model")
    print("Specifiche di normalizzazione del modello:")
    print(f"Normalizzazione: {processor.do_normalize}")
    print(f"Valori medi: {processor.image_mean}")
    print(f"Valori std: {processor.image_std}")
    print(f"Dimensione immagine: {processor.size}")
    
    visualize_preprocessing(processor, datasets_folder)
    validate_dataset_structure(dataset)
    
    return
    dataset = dataset.map((lambda x: preprocess(processor, x)), batched=True, remove_columns=["image"])
    test_dataloader = DataLoader(dataset["test"], batch_size=4, collate_fn=collate_fn)
    model.eval()

    all_preds, all_labels = evaluate(model, test_dataloader)
    mlflow_tracking(all_labels, all_preds, datasets_folder, folder_name)
    
if __name__ == "__main__":
    main()