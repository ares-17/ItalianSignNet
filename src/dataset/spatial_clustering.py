import os
import glob
import json
import pandas as pd
import numpy as np
from math import radians
from sklearn.cluster import DBSCAN
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path

"""
Questo script esegue il clustering dei cartelli stradali utilizzando l'algoritmo DBSCAN, tenendo conto delle seguenti considerazioni:

1. Vengono caricate le coordinate dei cartelli dai file GeoJSON presenti nella cartella "geojson_folder". 
   Per ciascuna feature, viene estratto l'ID (dal campo "properties.id") e le coordinate geografiche (in formato [lon, lat]).

2. Si legge il file "annotations.csv", che associa per ogni immagine l'etichetta del cartello. 
   Il nome del file è nel formato "${id_immagine_geojson}_${id_detection}.jpg". 
   Da questo viene estratto l'ID immagine (la parte prima dell'underscore).

3. I record risultanti (composti da ID immagine, etichetta e coordinate convertite in radianti) vengono raggruppati per etichetta.

4. Per ciascun gruppo (stessa etichetta) viene applicato DBSCAN con metrica "haversine" e un raggio massimo di 
   DBSCAN_DISTANCE (default 100 metri, convertiti in radianti). In questo modo vengono creati cluster di immagini 
   (cioè cartelli rilevati in immagini diverse) che mostrano lo stesso cartello.

5. Alla fine vengono generati report: per ogni etichetta, vengono stampati i cluster e, in report, il numero di cluster per 
   etichetta, la cardinalità del cluster più grande e la distribuzione dei cluster in base alla loro dimensione.
   Inoltre, viene calcolata la distribuzione totale dei cluster (tutte le etichette insieme).

6. Tutte le informazioni di log vengono scritte in un file denominato "dbscan_TIMESTAMP_eps_DBSCAN_DISTANCE.log" 
   nella stessa cartella dello script.

7. **Nuova funzionalità:** Le informazioni dei cluster vengono salvate in un file JSON che include:
    - La mappatura dei cluster per etichetta, con chiavi univoche (concatenazione di label e id DBSCAN)
    - Un report riepilogativo con il numero di cluster per label, la cardinalità del cluster più grande,
      la distribuzione per label e la distribuzione totale dei cluster.
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

load_dotenv(os.path.join(script_dir, '.env'))
BASE_DIR = Path(os.getenv("BASE_DIR"))
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
GEOJSON_FOLDER = os.path.join(BASE_DIR, Path(os.getenv("TEST_CASE_BASE_ROOT")), "geojson_folder")
ANNOTATIONS_CSV_FILE = os.path.join(BASE_DIR, Path(os.getenv("TEST_CASE_BASE_ROOT")), "annotations.csv")
MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", 1))

EARTH_RADIUS = 6371000
EPS_RAD = DBSCAN_DISTANCE / EARTH_RADIUS

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(logs_dir, f"spatial_clustering_{timestamp_str}_eps_{DBSCAN_DISTANCE}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename)]
)
logger = logging.getLogger(__name__)
logger.info("Inizio esecuzione clustering DBSCAN.")

def save_feature_coords(feature, geo_dict):
    props = feature.get("properties", {})
    feat_id = str(props.get("id"))
    coords = feature.get("geometry", {}).get("coordinates", None)
    if feat_id and coords:
        geo_dict[feat_id] = coords

# 1. Costruisci un dizionario per mappare l'id della feature a coordinate
def load_geojson_coordinates(geojson_folder):
    geo_dict = {}
    for filepath in glob.glob(os.path.join(geojson_folder, "*.geojson")):
        with open(filepath, "r") as f:
            data = json.load(f)
            [save_feature_coords(feature, geo_dict) for feature in data.get("features", [])]

    logger.info(f"Caricate {len(geo_dict)} coordinate da GeoJSON.")
    return geo_dict

def get_imageID_dataframe_from_csv(annotations_csv):
    df: pd.DataFrame = pd.read_csv(annotations_csv)
    df['image_id'] = df['filename'].apply(lambda x: str(x))
    return df

def load_annotations(annotations_csv, geo_dict) -> list:
    df = get_imageID_dataframe_from_csv(annotations_csv)
    records = []
    
    for _, row in df.iterrows():
        image_id = row["image_id"]
        label = row["feature"]
        coords = geo_dict.get(image_id.split("_")[0])
        if coords:
            lat_rad = radians(coords[1])
            lon_rad = radians(coords[0])
            records.append({"id": image_id, "label": label, "coords": [lat_rad, lon_rad], "raw_coords": [coords[1], coords[0]]})
            
    logger.info(f"Caricati {len(records)} record dalle annotazioni.")
    return records

def sanitize_param_name(name: str) -> str:
    """Sostituisce caratteri non validi con underscore"""
    return name.lower().replace('(', '_').replace(')', '_').replace('/', '_').replace(' ', '_')

def cluster_by_label(records, eps_rad):
    groups = {}
    for rec in records:
        groups.setdefault(rec["label"], []).append(rec)
    
    clusters_by_label = {}
    for label, recs in groups.items():
        coords = np.array([r["coords"] for r in recs])
        image_ids = [r["id"] for r in recs]
        
        if len(coords) == 0:
            continue
        
        clustering = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine").fit(coords)
        cluster_labels = clustering.labels_
        
        # Creiamo un dizionario con chiavi univoche: label + "_" + id_cluster
        clusters = {}
        for img_id, cluster_label in zip(image_ids, cluster_labels):
            normalized_label = sanitize_param_name(label)
            unique_cluster_id = f"{normalized_label}_{cluster_label}"
            clusters.setdefault(unique_cluster_id, []).append(img_id)
        clusters_by_label[normalized_label] = clusters
    return clusters_by_label

def aggregate_all_records_coords(records):
    """Aggrega record in un dizionario con chiavi ID e coordinate."""
    return {rec['id']: {'id': rec['id'], 'radius_coords': rec['coords'], 'coords': rec['raw_coords']} for rec in records}

def print_and_report_clusters(clusters_by_label):
    report = {}
    total_distribution = {}  # Distribuzione totale per cardinalità (tutte le etichette)
    
    for label, clusters in clusters_by_label.items():
        logger.info(f"Etichetta: {label}")
        num_clusters = len(clusters)
        largest_cluster_size = 0
        # Distribuzione per questa etichetta
        distribution = {}
        for cluster_key, img_ids in clusters.items():
            cluster_size = len(img_ids)
            logger.info(f"  Cluster {cluster_key}: {cluster_size} immagini -> {img_ids}")
            largest_cluster_size = max(largest_cluster_size, cluster_size)
            distribution[cluster_size] = distribution.get(cluster_size, 0) + 1
            
            # Aggiorna la distribuzione totale
            total_distribution[cluster_size] = total_distribution.get(cluster_size, 0) + 1
            
        report[label] = {
            "num_clusters": num_clusters,
            "largest_cluster_size": largest_cluster_size,
            "cluster_distribution": distribution
        }
        
    # Aggiungi la distribuzione totale al report
    report["total_cluster_distribution"] = total_distribution
    
    logger.info("Report finale:")
    for label, stats in report.items():
        if label != "total_cluster_distribution":
            logger.info(f"Etichetta: {label} | Numero cluster: {stats['num_clusters']} | Cardinalita del cluster piu grande: {stats['largest_cluster_size']}")
            logger.info(f"Distribuzione dei cluster: {stats['cluster_distribution']}")
    logger.info(f"Distribuzione totale dei cluster: {total_distribution}")
    
    return report

def convert_keys(d):
    """
    Converte le chiavi di un dizionario in int nativi se sono di tipo numpy.int64.
    Se il valore è un dizionario, la funzione viene applicata ricorsivamente.
    """
    new_d = {}
    for key, value in d.items():
        try:
            new_key = int(key)
        except (ValueError, TypeError):
            new_key = key
        if isinstance(value, dict):
            new_d[new_key] = convert_keys(value)
        else:
            new_d[new_key] = value
    return new_d

def save_clusters_to_json(clusters_by_label, report, coordinates_dict, output_path):
    """
    Salva le informazioni dei cluster in un file JSON.
    Il file conterrà sia la mappatura dei cluster per etichetta sia il report riepilogativo.
    """
    clusters_by_label_serializable = {
        label: convert_keys(clusters)
        for label, clusters in clusters_by_label.items()
    }
    
    data_to_save = {
        "clusters_by_label": clusters_by_label_serializable,
        "coordinates": coordinates_dict,
        "report": report
    }
    
    with open(output_path, "w") as f:
        json.dump(data_to_save, f, indent=2)
    
    logger.info(f"Informazioni dei cluster salvate in {output_path}")

def main():
    geo_dict = load_geojson_coordinates(GEOJSON_FOLDER)
    records = load_annotations(ANNOTATIONS_CSV_FILE, geo_dict)
    
    if not records:
        logger.error("Nessun record trovato. Verifica che gli ID nelle annotazioni corrispondano ai GeoJSON.")
        return
    
    clusters_by_label = cluster_by_label(records, EPS_RAD)
    coordinates_dict = aggregate_all_records_coords(records)
    report = print_and_report_clusters(clusters_by_label)
    
    output_filename = os.path.join(logs_dir, f"spatial_clustering_{timestamp_str}_eps_{DBSCAN_DISTANCE}.json")
    save_clusters_to_json(clusters_by_label, report, coordinates_dict, output_filename)
    
    logger.info("Esecuzione clustering DBSCAN completata.")

if __name__ == "__main__":
    main()
