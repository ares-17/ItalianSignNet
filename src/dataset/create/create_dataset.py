import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import shutil
from pathlib import Path
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR", "C:/Users/aschiavo/Documents/personale/ItalianSignNet")

current_directory = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(BASE_DIR, 'testing', 'bboxes_18_02')
CLUSTER_JSON_PATH = os.path.join(BASE_DIR, 'src', 'dataset','logs', 'dbscan_clusters_20250325_180404_eps_100.json')
OUTPUT_DIR = current_directory
TEST_SIZE = 0.2
VAL_SIZE = 0.1
SEED = 42

# Caricamento dei dati
with open(CLUSTER_JSON_PATH) as f:
    clustersInfo = json.load(f)

clusters: dict = clustersInfo['clusters_by_label']
metadata = pd.read_csv(f"{DATA_ROOT}/annotations.csv")

def assegna_cluster_id(metadata, clusters):
    cluster_id_map = {}
    for _, clusters_by_category in clusters.items():
        for cluster_name, detections in clusters_by_category.items():
            for filename in detections:
                cluster_id_map[filename] = cluster_name

    metadata['cluster_id'] = metadata['filename'].map(cluster_id_map)
    return metadata

metadata = assegna_cluster_id(metadata, clusters)
print(metadata)

# Split stratified mantenendo i cluster intatti
splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE + VAL_SIZE, random_state=SEED)
train_idx, temp_idx = next(splitter.split(metadata, groups=metadata['cluster_id']))

# Split ulteriore per validation/test
splitter_val_test = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE/(TEST_SIZE + VAL_SIZE), random_state=SEED)
val_idx, test_idx = next(splitter_val_test.split(metadata.iloc[temp_idx], groups=metadata.iloc[temp_idx]['cluster_id']))

# Assegnazione degli split
metadata['split'] = 'train'
metadata.iloc[temp_idx[val_idx], metadata.columns.get_loc('split')] = 'validation'
metadata.iloc[temp_idx[test_idx], metadata.columns.get_loc('split')] = 'test'


# Crea la struttura delle cartelle
for split in ['train', 'validation', 'test']:
    (Path(OUTPUT_DIR) / split).mkdir(parents=True, exist_ok=True)


# Sposta le immagini e aggiorna i percorsi
def organize_images(row):
    src_path = f"{DATA_ROOT}/resized_images/{row['filename']}"
    dest_path = f"{OUTPUT_DIR}/{row['split']}/{row['filename']}"
    shutil.copy(src_path, dest_path)
    return dest_path

metadata['image_path'] = metadata.apply(organize_images, axis=1)

# Salva il metadata in formato Parquet (più efficiente)
metadata.to_parquet(f"{OUTPUT_DIR}/metadata.parquet")

# Opzionale: Crea un file di esempio per il README
sample = metadata.sample(3)[['filename', 'label', 'bbox_coordinates']]
sample.to_markdown(f"{OUTPUT_DIR}/sample.md", index=False)

"""
# Tracking degli split e dei cluster
with mlflow.start_run():
    # Log delle distribuzioni
    mlflow.log_params({
        'train_clusters': metadata[metadata['split'] == 'train']['cluster_id'].nunique(),
        'test_clusters': metadata[metadata['split'] == 'test']['cluster_id'].nunique(),
        'total_clusters': metadata['cluster_id'].nunique()
    })
    
    # Log delle statistiche
    mlflow.log_metrics({
        'images_per_cluster_mean': metadata.groupby('cluster_id').size().mean(),
        'images_per_cluster_std': metadata.groupby('cluster_id').size().std()
    })
    
    # Log dell'intero dataset come artefatto
    mlflow.log_artifacts(OUTPUT_DIR, "dataset") """