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

"""
Questo script esegue il clustering dei cartelli stradali utilizzando l'algoritmo DBSCAN, tenendo conto delle seguenti considerazioni:

1. Vengono caricate le coordinate dei cartelli dai file GeoJSON presenti nella cartella "geojson_folder". 
   Per ciascuna feature, viene estratto l'ID (dal campo "properties.id") e le coordinate geografiche (in formato [lon, lat]).

2. Si legge il file "annotations.csv", che associa per ogni immagine l'etichetta del cartello. 
   Il nome del file è nel formato "${id_immagine_geojson}_${id_detection}.jpg". 
   Da questo viene estratto l'ID immagine (la parte prima dell'underscore).

3. Poiché in una stessa immagine possono essere presenti più rilevamenti (detections) dello stesso cartello, 
   e tali rilevamenti (cioè le coppie (immagine, etichetta)) non devono essere raggruppati in un cluster 
   (visto che rappresentano cartelli diversi presenti nella stessa immagine), si esegue una deduplicazione 
   per considerare **una sola occorrenza per coppia (immagine, etichetta)**.

4. I record risultanti (composti da ID immagine, etichetta e coordinate convertite in radianti) vengono raggruppati per etichetta.

5. Per ciascun gruppo (stessa etichetta) viene applicato DBSCAN con metrica "haversine" e un raggio massimo di 
   DBSCAN_DISTANCE (default 100 metri, convertiti in radianti). In questo modo vengono creati cluster di immagini 
   (cioè cartelli rilevati in immagini diverse) che mostrano lo stesso cartello.

6. Alla fine vengono generati report: per ogni etichetta, vengono stampati i cluster (solo quelli composti da più 
   di un'immagine) e, in report, il numero di cluster per etichetta e la cardinalità del cluster più grande.

7. Tutte le informazioni di log vengono scritte in un file denominato "dbscan_TIMESTAMP_eps_DBSCAN_DISTANCE.log" 
   nella stessa cartella dello script.
"""

load_dotenv()
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
GEOJSON_FOLDER = "testing/25_02/geojson_folder"
ANNOTATIONS_CSV = "testing/25_02/annotations.csv"
EARTH_RADIUS = 6371000
EPS_RAD = DBSCAN_DISTANCE / EARTH_RADIUS

script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(script_dir, f"dbscan_{timestamp_str}_eps_{DBSCAN_DISTANCE}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename)]
)
logger = logging.getLogger(__name__)
logger.info("Inizio esecuzione clustering DBSCAN.")

# 1. Costruisci un dizionario per mappare l'id della feature a coordinate
def load_geojson_coordinates(geojson_folder):
    geo_dict = {}
    for filepath in glob.glob(os.path.join(geojson_folder, "*.geojson")):
        with open(filepath, "r") as f:
            data = json.load(f)
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                feat_id = str(props.get("id"))
                coords = feature.get("geometry", {}).get("coordinates", None)
                if feat_id and coords:
                    geo_dict[feat_id] = coords
    logger.info(f"Caricate {len(geo_dict)} coordinate da GeoJSON.")
    return geo_dict

def load_annotations(annotations_csv, geo_dict):
    df = pd.read_csv(annotations_csv)
    # Estrae l'ID immagine dal filename e deduplica in base a (id, label)
    df['image_id'] = df['filename'].apply(lambda x: str(x).split("_")[0])
    df_unique = df.drop_duplicates(subset=["image_id", "feature"])
    
    records = []
    for _, row in df_unique.iterrows():
        image_id = row["image_id"]
        label = row["feature"]
        coords = geo_dict.get(image_id)
        if coords:
            lat_rad = radians(coords[1])
            lon_rad = radians(coords[0])
            records.append({"id": image_id, "label": label, "coords": [lat_rad, lon_rad]})
    logger.info(f"Caricati {len(records)} record unici dalle annotazioni.")
    return records

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
        
        clusters = {}
        for img_id, cluster_label in zip(image_ids, cluster_labels):
            clusters.setdefault(cluster_label, []).append(img_id)
        clusters_by_label[label] = clusters
    return clusters_by_label

def print_and_report_clusters(clusters_by_label):
    report = {}
    for label, clusters in clusters_by_label.items():
        logger.info(f"Etichetta: {label}")
        num_clusters = len(clusters)
        largest_cluster_size = 0
        for cluster_label, img_ids in clusters.items():
            # Se nel cluster ci sono immagini duplicate (lo stesso id), significa che quelle detections provengono dalla stessa immagine.
            # In tal caso, non consideriamo quel cluster per il report.
            if len(set(img_ids)) < len(img_ids):
                logger.info(f"  Cluster {cluster_label}: rilevamenti multipli dallo stesso immagine, ignorato per clustering")
            else:   
                if len(img_ids) > 1:
                    logger.info(f"  Cluster {cluster_label}: {len(img_ids)} immagini -> {img_ids}")
                largest_cluster_size = max(largest_cluster_size, len(img_ids))
        report[label] = {
            "num_clusters": num_clusters,
            "largest_cluster_size": largest_cluster_size
        }
    logger.info("Report finale:")
    for label, stats in report.items():
        if stats['largest_cluster_size'] > 1:
            logger.info(f"Etichetta: {label} | Numero cluster: {stats['num_clusters']} | Cardinalita del cluster piu grande: {stats['largest_cluster_size']}")
    logger.info("Tutte le etichette non riportate hanno cluster con al piu un elemento")
    return report

def main():
    geo_dict = load_geojson_coordinates(GEOJSON_FOLDER)
    records = load_annotations(ANNOTATIONS_CSV, geo_dict)
    
    if not records:
        logger.error("Nessun record trovato. Verifica che gli ID nelle annotazioni corrispondano ai GeoJSON.")
        return
    
    clusters_by_label = cluster_by_label(records, EPS_RAD)
    print_and_report_clusters(clusters_by_label)
    logger.info("Esecuzione clustering DBSCAN completata.")

if __name__ == "__main__":
    main()
