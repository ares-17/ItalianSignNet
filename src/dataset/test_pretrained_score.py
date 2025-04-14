import os
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
import pandas as pd
from datasets import load_dataset, ClassLabel
from typing import Dict, Any
import numpy as np
import logging
from datetime import datetime

REPO_MODEL: str = "bazyl/gtsrb-model"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# use or create log directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# logger definition
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(LOGS_DIR, f"test_pretrained_{TIMESTAMP}_model_{REPO_MODEL.replace('/', '_')}.log"))]
)
LOGGER = logging.getLogger(__name__)

def get_latest_dataset_folder() -> Tuple[str, str]:
    """
    Returns the path of the latest dataset folder within the artifacts directory.
    """
    BASE_DIR = os.getenv("BASE_DIR", "")
    artifacts_dir = os.path.join(BASE_DIR, "src", "dataset", "artifacts")
    
    all_subdirs = [
        d for d in os.listdir(artifacts_dir) if os.path.isdir(os.path.join(artifacts_dir, d))
    ]
    if not all_subdirs:
        raise ValueError("No dataset folder found in the artifacts directory")
    
    newest = sorted(all_subdirs)[-1]
    LOGGER.info(f"Selected dataset: {newest}")
    return os.path.join(artifacts_dir, newest), newest

def preprocess(
    processor: ImageProcessingMixin, examples: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Preprocesses a dataset sample by extracting pixel values and labels in a format suitable for the model.
    """
    inputs = processor(images=examples["image"], return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"].squeeze(),
        "labels": examples["label"],
    }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Collation function for the DataLoader.
    """
    return {
        "pixel_values": stack([tensor(x["pixel_values"]) for x in batch]),
        "labels": tensor([x["labels"] for x in batch]),
    }

def evaluate(
    model: PreTrainedModel, test_dataloader: DataLoader
) -> Tuple[List[int], List[int], List[int]]:
    """
    Evaluates the model on the test set.
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
    LOGGER.info("\nAccuracy per class:")
    for cls in present_classes:
        accuracy = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
        LOGGER.info(f"Class {cls}: {accuracy:.4f} ({class_correct[cls]}/{class_total[cls]})")
    
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
    Logs metrics and artifacts to MLflow, including detailed reports.
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
    
    LOGGER.info("Metrics logged successfully:")
    LOGGER.info(f"Accuracy: {accuracy:.4f}")
    LOGGER.info(f"Precision: {precision:.4f}")
    LOGGER.info(f"Recall: {recall:.4f}")
    LOGGER.info(f"F1-Score: {f1:.4f}")

def log_dataset_test_structure(dataset: Dict[str, Any]) -> None:
    """
    Checks that the dataset has the correct structure.
    """
    if "test" not in dataset:
        raise ValueError("The dataset must contain a 'test' split")
    
    test_labels = set(example["label"] for example in dataset["test"])
    
    LOGGER.info(f"Number of classes in the test set: {len(test_labels)}")
    
def load_dataset_with_features(data_dir: str, metadata_path: str) -> DatasetDict:
    """
    Loads the dataset with features and applies label correction using metadata.
    Only the 'test' split is loaded into memory.
    """
    metadata = pd.read_parquet(metadata_path)
    filename_to_label = dict(zip(metadata['filename'], metadata['feature_index'].astype(int)))
    
    features = Features({
        'image': Image(),
        'label': Value('int64')
    })
    
    # Load only the test split of the dataset
    test_dataset = load_dataset(
        "imagefolder",
        data_dir=data_dir,
        drop_labels=True,
        features=features,
        split="test"
    )
    
    def add_right_labels(example: Dict[str, Any]) -> Dict[str, Any]:
        filename = os.path.basename(example['image'].filename)
        example['label'] = filename_to_label[filename]
        return example
    
    test_dataset = test_dataset.map(add_right_labels)
    test_dataset = test_dataset.cast_column("label", ClassLabel(num_classes=43))
    
    return {"test": test_dataset}

def log_processor_info(processor):
    LOGGER.info("Model normalization specifications:")
    LOGGER.info(f"Normalization: {processor.do_normalize}")
    LOGGER.info(f"Mean values: {processor.image_mean}")
    LOGGER.info(f"Standard deviation values: {processor.image_std}")
    LOGGER.info(f"Image size: {processor.size}")

def main() -> None:    
    load_dotenv()
    datasets_folder, folder_name = get_latest_dataset_folder()
    metadata_path = os.path.join(datasets_folder, "metadata.parquet")
    
    dataset = load_dataset_with_features(
        data_dir=datasets_folder,
        metadata_path=metadata_path
    )
    processor = AutoImageProcessor.from_pretrained(REPO_MODEL)
    
    log_processor_info(processor)
    log_dataset_test_structure(dataset)
    
    model: PreTrainedModel = AutoModelForImageClassification.from_pretrained(REPO_MODEL)
    model.eval()
    
    dataset["test"] = dataset["test"].map(lambda x: preprocess(processor, x), batched=True, remove_columns=["image"])
    test_dataloader = DataLoader(dataset["test"], batch_size=4, collate_fn=collate_fn)
    
    all_preds, all_labels, present_classes = evaluate(model, test_dataloader)
    mlflow_tracking(all_labels, all_preds, datasets_folder, folder_name, present_classes)

if __name__ == "__main__":
    main()
