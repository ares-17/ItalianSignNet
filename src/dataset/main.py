from dotenv import load_dotenv
import os
import spatial_clustering
import create_dataset
import dbscan_reports
import augmentations
import requests
from requests.exceptions import RequestException

def check_mlflow_running(port=5000):
    try:
        response = requests.get(f"http://localhost:{port}")
        if response.status_code == 200:
            return True
        else:
            return False
    except RequestException as e:
        print(f"MLFlow error on {port}: {e}")
        return False

def main():
    load_dotenv()
    apply_augmentations = os.getenv("APPLY_AUGMENTATIONS", True)

    if not check_mlflow_running():
        print("Interrupting execution: MlFlow is not running")
        return

    spatial_clustering.main()
    create_dataset.main()
    dbscan_reports.main()
    
    if apply_augmentations:
        augmentations.main()


if __name__ == "__main__":
    main()