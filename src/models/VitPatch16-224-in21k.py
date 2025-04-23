from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    PreTrainedModel,
)
from vitpatch_utils import (VitPatchUtils, get_latest_dataset_folder)
from datetime import datetime

REPO_MODEL = "bazyl/gtsrb-model"

def main() -> None:
    start_time = datetime.now()        
    processor = AutoImageProcessor.from_pretrained(REPO_MODEL)
    model: PreTrainedModel = AutoModelForImageClassification.from_pretrained(REPO_MODEL)
    model.eval()
    
    dataset_folder, folder_name = get_latest_dataset_folder()
    
    utils = VitPatchUtils(REPO_MODEL, model, processor, dataset_folder, folder_name)
    utils.load_local_dataset()
    utils.log_info()
    utils.evaluate_in_mlflow()
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

if __name__ == "__main__":
    main()