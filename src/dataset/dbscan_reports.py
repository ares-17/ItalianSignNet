import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(script_dir, "reports")
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv(os.path.join(script_dir, '.env'))
GEOJSON_FOLDER = Path(os.getenv("GEOJSON_FOLDER"))
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
italy_bbox = tuple(map(float, os.getenv("ITALY_HEATMAP_BBOX").split(',')))

def load_geojson_coordinates(geojson_folder):
    """
    Costruisce un dizionario in cui le chiavi sono gli ID (convertiti in stringa) 
    e i valori sono le coordinate [lon, lat] estratte dai file GeoJSON.
    """
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
    return geo_dict

def compute_cluster_center(cluster_ids, geo_dict):
    """
    Calcola il centro (latitudine media e longitudine media) del cluster
    usando le coordinate associate agli ID contenuti in cluster_ids.
    """
    lats = []
    lons = []
    for img_id in cluster_ids:
        coord = geo_dict.get(img_id)
        if coord:
            lon, lat = coord
            lats.append(lat)
            lons.append(lon)
    if lats and lons:
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        return center_lat, center_lon
    return None

def heatmap(data):
    geojson_folder = os.path.join(script_dir, GEOJSON_FOLDER)
    clusters_by_label = data.get("clusters_by_label", {})
    geo_dict = load_geojson_coordinates(geojson_folder)
    
    centers = []
    for label, clusters in clusters_by_label.items():
        for cluster_id, image_ids in clusters.items():
            center = compute_cluster_center(image_ids, geo_dict)
            if center:
                centers.append(center)
    centers = np.array(centers)
    
    plt.figure(figsize=(10,8))
    if centers.size > 0:
        lats = centers[:,0]
        lons = centers[:,1]
        hb = plt.hexbin(lons, lats, gridsize=50, extent=(italy_bbox[1], italy_bbox[3], italy_bbox[0], italy_bbox[2]),
                        cmap='YlOrRd', mincnt=1)
        plt.colorbar(hb, label="Numero di cluster")
    plt.title("Heatmap dei cluster sulla mappa dell'Italia")
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    plt.plot([italy_bbox[1], italy_bbox[3], italy_bbox[3], italy_bbox[1], italy_bbox[1]],
             [italy_bbox[0], italy_bbox[0], italy_bbox[2], italy_bbox[2], italy_bbox[0]],
             color='blue', linewidth=2, label="Bounding Box Italia")
    plt.legend()
    heatmap_file = os.path.join(reports_dir, f"heatmap_clusters_{timestamp_str}_eps_{DBSCAN_DISTANCE}.png")
    plt.savefig(heatmap_file, dpi=300)
    plt.close()

def bar_graph(report_data):
    total_distribution = report_data.get("total_cluster_distribution", {})
    if total_distribution:
        # Convertiamo le chiavi in interi e ordiniamo la distribuzione per cardinalità
        dist_items: list[tuple[int, int]] = [(int(k), v) for k, v in total_distribution.items()]
        dist_items = sorted(dist_items, key=lambda x: x[0])
        cardinalities, counts = zip(*dist_items)
        plt.figure(figsize=(8,6))
        plt.bar(cardinalities, counts, color='skyblue')
        plt.xlabel("Cardinalità del cluster")
        plt.ylabel("Numero di cluster")
        plt.title("Distribuzione totale dei cluster per cardinalità")
        plt.xticks(cardinalities)
        bar_chart_file = os.path.join(reports_dir, f"bar_chart_cluster_distribution_{timestamp_str}_eps_{DBSCAN_DISTANCE}.png")
        plt.savefig(bar_chart_file, dpi=300)
        plt.close()

def labels_by_bar_graph(report_data):
    labels = []
    num_clusters = []
    for label, stats in report_data.items():
        if label != "total_cluster_distribution":
            labels.append(label)
            num_clusters.append(stats.get("num_clusters", 0))
    if labels:
        plt.figure(figsize=(10,6))
        plt.bar(labels, num_clusters, color='lightgreen')
        plt.xlabel("Etichetta")
        plt.ylabel("Numero di cluster")
        plt.title("Numero di cluster per etichetta")
        plt.xticks(rotation=45, ha="right")
        clusters_per_label_file = os.path.join(reports_dir, f"bar_chart_clusters_per_label_{timestamp_str}_eps_{DBSCAN_DISTANCE}.png")
        plt.tight_layout()
        plt.savefig(clusters_per_label_file, dpi=300)
        plt.close()

def main():
    os.makedirs(reports_dir, exist_ok=True)

    dbscan_file = "dbscan_clusters_20250320_193435_eps_100.json"
    logs_dir = os.path.join(script_dir, "logs")
    clusters_file = os.path.join(logs_dir, dbscan_file)

    with open(clusters_file, "r") as f:
        data = json.load(f)

    report_data = data.get("report", {})

    heatmap(data)
    bar_graph(report_data)
    labels_by_bar_graph(report_data)

if __name__ == "__main__":
    main()
