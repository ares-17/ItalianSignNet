import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.data
import mlflow.data.pandas_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import shutil

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
DATA_SOURCE_ROOT = os.path.join(BASE_DIR, os.getenv("TEST_CASE_BASE_ROOT"))
CLUSTER_JSON_PATH = os.path.join(BASE_DIR, 'src', 'dataset', 'logs', os.getenv("DBSCAN_JSON_CLUSTERS"))
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
OUTPUT_DIR = os.path.join(BASE_DIR, 'src', 'dataset', 'artifacts', f'dataset_{datetime.now().strftime("%Y%m%d_%H%M%S")}_eps_${DBSCAN_DISTANCE}')

mlflow.set_tracking_uri('http://localhost:5000')

def assign_cluster_id(metadata: pd.DataFrame, clusters: dict) -> pd.DataFrame:
    """
    Assegna un cluster ID a ciascun sample del metadata basandosi sulle informazioni dei cluster.

    Args:
        metadata: DataFrame contenente i metadati delle immagini
        clusters: Dizionario con la mappatura dei cluster per categoria

    Returns:
        DataFrame arricchito con la colonna 'cluster_id'
    """
    cluster_id_map = {}
    for _, clusters_by_category in clusters.items():
        for cluster_name, detections in clusters_by_category.items():
            for filename in detections:
                cluster_id_map[filename] = cluster_name

    metadata['cluster_id'] = metadata['filename'].map(cluster_id_map)
    return metadata

def save_to_parquet(metadata: pd.DataFrame) -> None:
    output_path = os.path.join(OUTPUT_DIR, "metadata.parquet")
    metadata.to_parquet(output_path)

def organize_images(row: pd.Series) -> str:
    """
    Organizza le immagini nelle directory corrispondenti allo split assegnato.

    Args:
        row: Riga del DataFrame contenente le informazioni dell'immagine

    Returns:
        Percorso di destinazione dell'immagine copiata
    """
    src_path = os.path.join(DATA_SOURCE_ROOT, "resized_images", row['filename'])
    dest_path = os.path.join(OUTPUT_DIR, row['split'], row['filename'])
    shutil.copy(src_path, dest_path)
    return dest_path

def create_folder_sets() -> None:
    for split in ['train', 'validation', 'test']:
        dir_path = os.path.join(OUTPUT_DIR, split)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def split_dataset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Suddivide il dataset in train, validation e test mantenendo l'integrità dei cluster.

    Args:
        metadata: DataFrame contenente i metadati da suddividere

    Returns:
        DataFrame con la colonna aggiuntiva 'split' che indica la partizione
    """
    seed = int(os.getenv("SEED_GROUP_SHUFFLE_DATASET", 42))
    test_size = float(os.getenv("TEST_SIZE_DATASET", 0.2))
    val_size = float(os.getenv("VAL_SIZE_DATASET", 0.1))

    # Split stratified mantenendo i cluster intatti
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size + val_size, random_state=seed)
    _, temp_idx = next(splitter.split(metadata, groups=metadata['cluster_id']))

    # Split ulteriore per validation/test
    splitter_val_test = GroupShuffleSplit(n_splits=1, test_size=val_size/(test_size + val_size), random_state=seed)
    val_idx, test_idx = next(splitter_val_test.split(metadata.iloc[temp_idx], groups=metadata.iloc[temp_idx]['cluster_id']))

    # Assegnazione degli split
    metadata['split'] = 'train'
    metadata.iloc[temp_idx[val_idx], metadata.columns.get_loc('split')] = 'validation'
    metadata.iloc[temp_idx[test_idx], metadata.columns.get_loc('split')] = 'test'

    return metadata

def copy_images_to_output_path(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Organizza i samples nelle directory.

    Args:
        metadata: DataFrame contenente i metadati

    Returns:
        DataFrame con la colonna aggiuntiva 'image_path'
    """
    metadata.apply(organize_images, axis=1)
    return metadata

def log_dataset_info(metadata: pd.DataFrame, cluster_report: dict) -> None:
    """
    Log dataset information to MLflow with enhanced tracking capabilities.
    
    Args:
        metadata: DataFrame contenente le informazioni sul dataset.
        cluster_report: Dizionario con statistiche relative ai cluster.
    """
    # Crea un dataset a partire dal DataFrame
    dataset = mlflow.data.from_pandas(
        metadata,
        source=DATA_SOURCE_ROOT,
        name=f"ItalianTrafficSignDataset_${DBSCAN_DISTANCE}",
        targets="feature"
    )

    with mlflow.start_run(run_name="full_experiment"):
        with mlflow.start_run(run_name="dataset_creation", nested=True):
            mlflow.log_input(dataset, context="full_dataset")

            for split_name in ['train', 'validation', 'test']:
                split_df = metadata[metadata['split'] == split_name]
                if not split_df.empty:
                    split_dataset = mlflow.data.from_pandas(
                        split_df,
                        source=dataset.source,
                        name=f"ItalianTrafficSigns_{split_name}",
                        targets="feature"
                    )
                    mlflow.log_input(split_dataset, context=f"{split_name}_data")
                    
                    mlflow.log_metric(
                        f"{split_name}_clusters", 
                        split_df['cluster_id'].nunique()
                    )

            # Log metriche e statistiche globali del dataset
            mlflow.log_metrics({
                'unique_clusters_total': metadata['cluster_id'].nunique(),
                'images_per_cluster_mean': metadata.groupby('cluster_id').size().mean()
            })

            # Log distribuzione delle feature (classi) e report sui cluster
            feature_dist = metadata['feature'].value_counts().to_dict()
            mlflow.log_dict(feature_dist, "feature_distribution.json")
            mlflow.log_dict(cluster_report, "cluster_report.json")

            # Imposta dei tag importanti per il dataset
            mlflow.set_tags({
                "date": datetime.now().strftime("%Y-%m-%d"),
                "version": 0.1,
                "eps": DBSCAN_DISTANCE,
                "folder": os.getenv("TEST_CASE_BASE_ROOT"),
                "json_spatial_clustering": CLUSTER_JSON_PATH
            })

            mlflow.log_artifact(CLUSTER_JSON_PATH)

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(CLUSTER_JSON_PATH) as f:
        clustersInfo: dict = json.load(f)
    
    create_folder_sets()

    clusters: dict = clustersInfo['clusters_by_label']
    metadata: pd.DataFrame = pd.read_csv(os.path.join(DATA_SOURCE_ROOT, "annotations.csv"))
    metadata = assign_cluster_id(metadata, clusters)
    metadata = split_dataset(metadata)
    metadata = copy_images_to_output_path(metadata)

    save_to_parquet(metadata)
    log_dataset_info(metadata, clustersInfo['report'])

if __name__ == "__main__":
    main()