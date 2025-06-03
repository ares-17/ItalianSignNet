import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import mlflow
import mlflow.data
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import shutil

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
DATA_SOURCE_ROOT = os.path.join(BASE_DIR, os.getenv("TEST_CASE_BASE_ROOT"))
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
OUTPUT_DIR = os.path.join(BASE_DIR, 'src', 'dataset', 'artifacts', f'dataset_{TIMESTAMP}_eps_{DBSCAN_DISTANCE}')
LABEL_INDEX_FILE = os.path.join(BASE_DIR, 'src', 'utils', 'signnames.csv')

mlflow.set_tracking_uri('http://localhost:5000')

def get_latest_json() -> str:
    logs_dir = os.path.join(BASE_DIR, 'src', 'dataset', 'logs')
    json_files = [
        f for f in os.listdir(logs_dir)
        if f.endswith('.json') and os.path.isfile(os.path.join(logs_dir, f))
    ]
    
    if not json_files:
        raise ValueError(f"Nessun file JSON trovato nella cartella {logs_dir}")
    
    latest_json = sorted(json_files)[-1]
    json_path = os.path.join(logs_dir, latest_json)
    return json_path

CLUSTER_JSON_PATH = get_latest_json()

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

def sanitize_param_name(name: str) -> str:
    """Sostituisce caratteri non validi con underscore"""
    return name.lower().replace('(', '_').replace(')', '_').replace('/', '_').replace(' ', '_')

def precreate_class_folders(split: str, base_dir: str) -> None:
    """
    Crea preventivamente tutte le cartelle delle classi per ogni split.
    """
    for class_id in range(43):
        dir_path = os.path.join(base_dir, split, f"{class_id:02d}")
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def organize_images(row: pd.Series) -> str:
    """
    Organizza le immagini nelle directory corrispondenti allo split assegnato.
    """
    
    src_path = os.path.join(DATA_SOURCE_ROOT, "resized_images", row['filename'])
    dest_dir = os.path.join(OUTPUT_DIR, row['split'], f"{row['feature_index']:02d}")
    dest_path = os.path.join(dest_dir, row['filename'])
    
    shutil.copy(src_path, dest_path)
    return dest_path

def create_folder_sets() -> None:
    for split in ['train', 'validation', 'test']:
        dir_path = os.path.join(OUTPUT_DIR, split)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def aggregate_too_small_cluster_val_test(cluster_labels, temp_idx):
    temp_clusters = cluster_labels.iloc[temp_idx].copy()
    temp_label_counts = temp_clusters['feature_index'].value_counts()
    too_small_classes = temp_label_counts[temp_label_counts < 2].index.tolist()

    if too_small_classes:
        print(f"Classi troppo piccole per il secondo split: {too_small_classes}")
        temp_clusters = temp_clusters[~temp_clusters['feature_index'].isin(too_small_classes)]

    return temp_clusters

def split_dataset(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Suddivide il dataset in train/validation/test mantenendo:
    1. Cluster interi nello stesso split
    2. Distribuzione delle etichette bilanciata
    3. Gestione di edge case critici
    4. Solo classi con almeno 10 cartelli
    """
    seed = int(os.getenv("SEED_GROUP_SHUFFLE_DATASET", 42))
    test_size = float(os.getenv("TEST_SIZE_DATASET", 0.2))
    val_size = float(os.getenv("VAL_SIZE_DATASET", 0.1))

    # 0. Filtro: considera solo classi con almeno 10 cartelli totali
    label_counts_total = metadata['feature_index'].value_counts()
    valid_labels = label_counts_total[label_counts_total >= 10].index.tolist()
    metadata = metadata[metadata['feature_index'].isin(valid_labels)]

    # 1. Preparazione dati a livello di cluster
    cluster_labels = metadata.drop_duplicates('cluster_id')[['cluster_id', 'feature_index']]
    
    # 2. Controllo edge case: numero minimo di cluster per etichetta
    label_counts = cluster_labels['feature_index'].value_counts()
    problematic_labels = label_counts[label_counts < 3].index.tolist()
    
    if problematic_labels:
        print(f"Etichette {problematic_labels} hanno meno di 3 cluster. Impossibile garantire split bilanciato.")
        cluster_labels = cluster_labels.loc[~cluster_labels['feature_index'].isin(problematic_labels)]

    # 3. Split stratificato a livello di cluster
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size + val_size,
        random_state=seed
    )
    
    train_idx, temp_idx = next(sss.split(
        X=cluster_labels[['cluster_id']], 
        y=cluster_labels['feature_index']
    ))
    
    # 4. Split val/test
    sss_val_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size / (test_size + val_size),
        random_state=seed
    )
    
    val_test_cluster_labels_normalized = \
        aggregate_too_small_cluster_val_test(cluster_labels, temp_idx)

    val_idx, test_idx = next(sss_val_test.split(
        X=val_test_cluster_labels_normalized[['cluster_id']],
        y=val_test_cluster_labels_normalized['feature_index']
    ))

    # 5. Mappatura dei cluster agli split
    splits = {
        'train': cluster_labels.iloc[train_idx]['cluster_id'],
        'validation': val_test_cluster_labels_normalized.iloc[val_idx]['cluster_id'],
        'test': val_test_cluster_labels_normalized.iloc[test_idx]['cluster_id']
    }

    metadata['split'] = 'train'
    for split_name, clusters in splits.items():
        metadata.loc[metadata['cluster_id'].isin(clusters), 'split'] = split_name

    return metadata

def verify_splits(metadata):
    return (
        metadata
        .groupby(['split', 'feature_index'])
        .size()
        .unstack()
        .apply(lambda x: x/x.sum(), axis=1)
    )

def copy_images_to_output_path(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Organizza i samples nelle directory.

    Args:
        metadata: DataFrame contenente i metadati

    Returns:
        DataFrame con la colonna aggiuntiva 'image_path'
    """
    
    precreate_class_folders('train', OUTPUT_DIR)
    precreate_class_folders('test', OUTPUT_DIR)
    precreate_class_folders('validation', OUTPUT_DIR)
    metadata.apply(organize_images, axis=1)
    return metadata

def add_coordinates_to_dataframe(df: pd.DataFrame, coordinates):
    """Aggiunge una colonna 'coordinates' al DataFrame con le coordinate formattate come stringa."""
    df['coordinates'] = df['filename'].apply(
        lambda img_id: f"{coordinates[img_id]['coords'][0]},{coordinates[img_id]['coords'][1]}"
    )
    return df

def df_add_label_index_column(metadata: pd.DataFrame) -> pd.DataFrame:
    pd_indexes = pd.read_csv(LABEL_INDEX_FILE)

    pd_indexes["SignName_lower"] = pd_indexes["SignName"].str.lower()
    metadata["feature_lower"] = metadata["feature"].str.lower()

    label_map = dict(zip(pd_indexes["SignName_lower"], pd_indexes["ClassId"]))

    metadata["feature_index"] = metadata["feature_lower"].map(label_map)
    # Drop temporary column
    metadata.drop(columns=["feature_lower"], inplace=True)

    if metadata["feature_index"].isnull().any():
        unmapped = metadata[metadata["feature_index"].isnull()]["feature"].unique()
        raise ValueError(f"Le seguenti feature non sono state trovate nel file label index (case-insensitive): {unmapped}")

    metadata["feature_index"] = metadata["feature_index"].astype(int)

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
        name=f"ItalianTrafficSignDataset_{DBSCAN_DISTANCE}",
        targets="feature"
    )

    with mlflow.start_run(run_name=f"dataset_creation_{TIMESTAMP}"):
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
    metadata = df_add_label_index_column(metadata)
    metadata = add_coordinates_to_dataframe(metadata, clustersInfo['coordinates'])
    metadata = assign_cluster_id(metadata, clusters)
    metadata = split_dataset(metadata)
    metadata = copy_images_to_output_path(metadata)

    save_to_parquet(metadata)
    log_dataset_info(metadata, clustersInfo['report'])

if __name__ == "__main__":
    main()