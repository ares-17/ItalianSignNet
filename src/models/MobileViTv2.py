from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from mobile_vitpatch_utils import (MobileVitPatchUtils, get_latest_dataset_folder)

REPO_MODEL="shehan97/mobilevitv2-1.0-imagenet1k-256"

def main() -> None:        
    processor = MobileViTImageProcessor.from_pretrained(REPO_MODEL)
    model = MobileViTV2ForImageClassification.from_pretrained(REPO_MODEL)
    model.eval()
    
    dataset_folder, folder_name = get_latest_dataset_folder()
    
    utils = MobileVitPatchUtils(REPO_MODEL, model, processor, dataset_folder, folder_name)
    utils.load_local_dataset()
    utils.log_info()
    utils.evaluate_in_mlflow()

if __name__ == "__main__":
    main()
