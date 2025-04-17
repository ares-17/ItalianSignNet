from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    PreTrainedModel,
)
from model_utils import (TestModel, get_latest_dataset_folder)

REPO_MODEL = "bazyl/gtsrb-model"

def main() -> None:        
    processor = AutoImageProcessor.from_pretrained(REPO_MODEL)
    model: PreTrainedModel = AutoModelForImageClassification.from_pretrained(REPO_MODEL)
    model.eval()
    
    dataset_folder, folder_name = get_latest_dataset_folder()
    
    utils = TestModel(REPO_MODEL, model, processor, dataset_folder, folder_name)
    utils.load_local_dataset()
    utils.log_info()
    utils.evaluate_in_mlflow()

if __name__ == "__main__":
    main()