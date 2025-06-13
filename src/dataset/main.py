from dotenv import load_dotenv
import os
import spatial_clustering
import create_dataset
import dbscan_reports
import augmentations
import requests
from requests.exceptions import RequestException

def check_mlflow_running(mlflow_endpoint: str | None):
    try:
        response = requests.get(mlflow_endpoint if mlflow_endpoint else "http://localhost:5000")
        if response.status_code == 200:
            return True
        else:
            return False
    except RequestException as e:
        print(f"MLFlow error on {port}: {e}")
        return False

def main():
    load_dotenv()
    mlflow_endpoint = os.getenv("MLFLOW_ENDPOINT", None)
    apply_augmentations = os.getenv("APPLY_AUGMENTATIONS", True)

    if not check_mlflow_running(mlflow_endpoint):
        print("Interrupting execution: MlFlow is not running")
        return

    spatial_clustering.main()
    create_dataset.main()
    
    if apply_augmentations:
        augmentations.main()
    
    dbscan_reports.main()

if __name__ == "__main__":
    main()