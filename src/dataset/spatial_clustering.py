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
Questo script esegue il clustering delle immagini dei cartelli stradali utilizzando l'algoritmo DBSCAN.
I passi compiuti sono i seguenti:

1. Carica le coordinate dei cartelli dai file GeoJSON presenti nella cartella "geojson_folder". 
   Per ciascuna feature, viene estratto l'ID (proveniente dal campo "properties.id") e le coordinate geografiche 
   (nell'ordine [lon, lat]).

2. Legge il file "annotations.csv", che contiene le associazioni tra il nome del file immagine e l'etichetta (feature).
   Il nome del file ha il formato "${id_immagine_geojson}_${id_detection}.jpg". Da questo viene estratto l'ID immagine 
   (la parte prima dell'underscore). Utilizzando questo ID, lo script cerca le coordinate corrispondenti nel dizionario 
   creato al punto 1.

3. Per ciascun record (composto da ID immagine, etichetta e coordinate convertite in radianti), vengono raggruppati i dati 
   per etichetta.

4. Per ogni gruppo di immagini (stessa etichetta), viene applicato DBSCAN con metrica "haversine" e un raggio massimo 
   di 100 metri (convertiti in radianti). In questo modo, vengono creati cluster di immagini che rappresentano lo stesso cartello.

5. Il risultato viene stampato a video: per ogni etichetta, vengono indicati solo i cluster che contengono più di una immagine.
   Inoltre, viene generata una reportistica che indica per ogni etichetta il numero di cluster creati e la cardinalità 
   (numero di immagini) del cluster più grande.
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
    handlers=[
        logging.FileHandler(log_filename)
    ]
)
logger = logging.getLogger(__name__)
logger.info("Inizio esecuzione clustering DBSCAN.")

# 1. Costruisci un dizionario per mappare l'id della feature a coordinate
def load_geojson_coordinates(geojson_folder):
    geo_dict = {}
    # Considera tutti i file .geojson nella cartella
    for filepath in glob.glob(os.path.join(geojson_folder, "*.geojson")):
        with open(filepath, "r") as f:
            data = json.load(f)
            # Per ogni feature, usa properties["id"] come chiave e geometry["coordinates"] come valore
            for feature in data.get("features", []):
                props = feature.get("properties", {})
                feat_id = str(props.get("id"))
                coords = feature.get("geometry", {}).get("coordinates", None)
                if feat_id and coords:
                    # Le coordinate in GeoJSON sono [lon, lat]
                    geo_dict[feat_id] = coords
    logger.info(f"Caricate {len(geo_dict)} coordinate da GeoJSON.")
    return geo_dict

# 2. Leggi il file annotations.csv e costruisci una lista di record con id, etichetta e coordinate
def load_annotations(annotations_csv, geo_dict):
    df = pd.read_csv(annotations_csv)
    records = []
    # Per ogni riga, il filename ha formato "${id_immagine_geojson}_${id_detection}.jpg"
    for _, row in df.iterrows():
        filename = str(row["filename"])
        label = row["feature"]
        # Estrae l'ID immagine: es. "1040331487377490" da "1040331487377490_191.jpg"
        image_id = filename.split("_")[0]
        # Cerca le coordinate nel dizionario
        coords = geo_dict.get(image_id)
        if coords:
            # Convertiamo le coordinate [lon, lat] in radianti (per DBSCAN con haversine, l'ordine è [lat, lon])
            lat_rad = radians(coords[1])
            lon_rad = radians(coords[0])
            records.append({"id": image_id, "label": label, "coords": [lat_rad, lon_rad]})
    logger.info(f"Caricati {len(records)} record dalle annotazioni.")
    return records

# 3. Per ciascuna etichetta, applica DBSCAN per raggruppare le immagini simili
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
        
        # Applica DBSCAN con metrica haversine
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
            if len(img_ids) > 1:
                logger.info(f"  Cluster {cluster_label}: {len(img_ids)} immagini -> {img_ids}")
            largest_cluster_size = max(largest_cluster_size, len(img_ids))
        report[label] = {
            "num_clusters": num_clusters,
            "largest_cluster_size": largest_cluster_size
        }
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
