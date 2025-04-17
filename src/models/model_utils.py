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
    PreTrainedModel,
)
import seaborn as sns
import pandas as pd
from datasets import load_dataset, ClassLabel
from typing import Dict, Any
import numpy as np
import logging
from datetime import datetime

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
    return os.path.join(artifacts_dir, newest), newest

class TestModel:
    def __init__(self, model_name: str, model: PreTrainedModel, processor: Any, datasets_folder: str, folder_name: str):
        self.repo_model: str = model_name
        self.logger = self._setup_logger()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = model
        self.processor = processor
        self.datasets_folder = datasets_folder 
        self.folder_name = folder_name 
        
    def _setup_logger(self) -> logging.Logger:
        """Initialize and return a logger with file handler."""
        BASE_DIR = os.getenv("BASE_DIR", "")
        logs_dir = os.path.join(BASE_DIR, "src", "models", "logs")
        os.makedirs(logs_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(logs_dir, f"test_pretrained_{timestamp}_model_{self.repo_model.replace('/', '_')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_file)]
        )
        return logging.getLogger(__name__)

    def _preprocess(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses a dataset sample by extracting pixel values and labels in a format suitable for the model.
        """
        inputs = self.processor(images=examples["image"], return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "labels": examples["label"],
        }

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        """
        Collation function for the DataLoader.
        """
        return {
            "pixel_values": stack([tensor(x["pixel_values"]) for x in batch]),
            "labels": tensor([x["labels"] for x in batch]),
        }

    def _evaluate(
        self,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Evaluates the model on the test set.
        """
        all_preds: List[int] = []
        all_labels: List[int] = []
        
        class_correct: List[int] = [0] * 43
        class_total: List[int] = [0] * 43

        with no_grad():
            for batch in self.test_dataloader:
                outputs = self.model(pixel_values=batch["pixel_values"])
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
        self.logger.info("\nAccuracy per class:")
        for cls in present_classes:
            accuracy = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            self.logger.info(f"Class {cls}: {accuracy:.4f} ({class_correct[cls]}/{class_total[cls]})")
        
        return all_preds, all_labels, present_classes

    def _plot_confusion_matrix(self, all_labels, all_preds, present_classes):
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

    def _mlflow_tracking(
        self, 
        all_labels: List[int],
        all_preds: List[int],
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
        
        with mlflow.start_run(run_name=f"{self.repo_model}_{self.folder_name}"):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
            mlflow.log_figure(self._plot_confusion_matrix(all_labels, all_preds, present_classes), "confusion_matrix.png")
            mlflow.log_param("model_name", self.repo_model)
            mlflow.log_param("dataset_version", os.path.basename(self.datasets_folder))
        
        self.logger.info("Metrics logged successfully:")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1-Score: {f1:.4f}")

    def _log_dataset_test_structure(self) -> None:
        """
        Checks that the dataset has the correct structure.
        """
        if "test" not in self.dataset:
            raise ValueError("The dataset must contain a 'test' split")
        
        test_labels = set(example["label"] for example in self.dataset["test"])
        
        self.logger.info(f"Number of classes in the test set: {len(test_labels)}")
        
    def _load_dataset_with_features(self, data_dir: str, metadata_path: str) -> DatasetDict:
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

    def _log_processor_info(self):
        self.logger.info("Model normalization specifications:")
        self.logger.info(f"Normalization: {self.processor.do_normalize}")
        self.logger.info(f"Mean values: {self.processor.image_mean}")
        self.logger.info(f"Standard deviation values: {self.processor.image_std}")
        self.logger.info(f"Image size: {self.processor.size}")

    def load_local_dataset(self) -> DataLoader:
        load_dotenv()
        metadata_path = os.path.join(self.datasets_folder, "metadata.parquet")
        
        self.dataset = self._load_dataset_with_features(
            data_dir=self.datasets_folder,
            metadata_path=metadata_path
        )
        
        self.dataset["test"] = self.dataset["test"].map(self._preprocess, batched=True, remove_columns=["image"])
        self.test_dataloader = DataLoader(self.dataset["test"], batch_size=4, collate_fn=self._collate_fn)
        
        return self.test_dataloader, self.folder_name

    def log_info(self):
        self._log_processor_info()
        self._log_dataset_test_structure()

    def evaluate_in_mlflow(self):
        all_preds, all_labels, present_classes = self._evaluate()
        self._mlflow_tracking(all_labels, all_preds, present_classes)