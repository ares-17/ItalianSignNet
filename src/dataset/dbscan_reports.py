import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import pandas as pd
import glob
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
reports_dir = os.path.join(script_dir, "reports")
os.makedirs(reports_dir, exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv(os.path.join(script_dir, '.env'))
BASE_DIR = os.getenv("BASE_DIR") or ''
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
ITALY_HEATMAP_BBOX = tuple(map(float, os.getenv("ITALY_HEATMAP_BBOX").split(',')))
PARQUET_FILE = os.path.join(script_dir, "data", "clustering_results.parquet")

def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    # Split coordinates into two columns
    df[['lon', 'lat']] = df['coordinates'].str.split(',', expand=True).astype(float)
    return df

def compute_cluster_centers(df):
    return df.groupby('cluster_id')[['lat', 'lon']].mean().reset_index()

def plot_heatmap(cluster_centers):
    plt.figure(figsize=(10, 8))
    if not cluster_centers.empty:
        lons = cluster_centers['lon']
        lats = cluster_centers['lat']
        
        hb = plt.hexbin(lats, lons, gridsize=50,
                        extent=(ITALY_HEATMAP_BBOX[1], ITALY_HEATMAP_BBOX[3],  # xmin, xmax (lon)
                               ITALY_HEATMAP_BBOX[0], ITALY_HEATMAP_BBOX[2]),  # ymin, ymax (lat)
                        cmap='YlOrRd', mincnt=1)
        
        plt.colorbar(hb, label="Numero di cluster")
    
    plt.title("Heatmap dei cluster sulla mappa dell'Italia")
    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")
    
    plt.xlim(6, 19)  # Longitudine approssimativa Italia
    plt.ylim(36, 48)  # Latitudine approssimativa Italia
    
    plt.savefig(os.path.join(reports_dir, f"heatmap_clusters_{timestamp_str}_eps_{DBSCAN_DISTANCE}.png"), dpi=300)
    plt.close()

def plot_cluster_cardinality(df):
    cluster_sizes = df.groupby('cluster_id').size().value_counts().sort_index()
    plt.figure(figsize=(8, 6))
    plt.bar(cluster_sizes.index, cluster_sizes.values, color='skyblue')
    plt.xlabel("Cardinalità del cluster")
    plt.ylabel("Numero di cluster")
    plt.title("Distribuzione totale dei cluster per cardinalità")
    plt.xticks(cluster_sizes.index)
    plt.savefig(os.path.join(reports_dir, f"bar_chart_cluster_distribution_{timestamp_str}_eps_{DBSCAN_DISTANCE}.png"), dpi=300)
    plt.close()

def plot_clusters_per_label(df):
    clusters_per_label = df.groupby('feature')['cluster_id'].nunique()
    plt.figure(figsize=(10, 6))
    clusters_per_label.plot(kind='bar', color='lightgreen')
    plt.xlabel("Etichetta")
    plt.ylabel("Numero di cluster")
    plt.title("Numero di cluster per etichetta")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"bar_chart_clusters_per_label_{timestamp_str}_eps_{DBSCAN_DISTANCE}.png"), dpi=300)
    plt.close()

def get_input_dataset_or_latest(dataset_name=None) -> str:
    base_dir_dataset = Path(BASE_DIR, 'src', 'dataset', 'artifacts')

    if dataset_name:
        return os.path.join(base_dir_dataset, dataset_name)

    dataset_folders = glob.glob(os.path.join(base_dir_dataset, "dataset_*"))
    if not dataset_folders:
        raise FileNotFoundError("Nessun dataset trovato")

    dataset_folders.sort(reverse=True)
    return dataset_folders[0]

def plot_features_by_quartile(df):
    plt.figure(figsize=(10, 6))
    pd.crosstab(df['feature'], df['income_quartile']).plot(kind='bar', stacked=True)
    plt.title("Distribuzione delle feature per quartile di reddito")
    plt.xlabel("Feature")
    plt.ylabel("Conteggio")
    plt.xticks(rotation=45, ha="right")
    plt.xticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"features_by_quartile_{timestamp_str}.png"), dpi=300)
    plt.close()


import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_quartile_counts_by_area(df):
    plt.figure(figsize=(10, 6))
    
    ax = sns.countplot(
        data=df, 
        x='area', 
        hue='income_quartile',
        order=['nord', 'centre', 'sud'],             # Ordine macro-aree N→S
        hue_order=['Q1', 'Q2', 'Q3', 'Q4'],          # Ordine quartili Q1→Q4
        palette='Set2'                               # Colori vivaci
    )
    
    plt.title('Conteggio dei quartili di reddito per macro-area')
    plt.xlabel('Macro-area')
    plt.ylabel('Conteggio')
    plt.legend(title='Quartile', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Etichette sopra le barre
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(os.path.join(reports_dir, f"quartile_counts_by_area_{timestamp_str}.png"), dpi=300)
    plt.close()


def main(dataset_name = None):
    """
    Funzione principale. Può essere richiamata da script esterni oppure da riga di comando.
    
    :param dataset_name: Nome della cartella del dataset (es: 'dataset_20240624') oppure None per usare l'ultimo disponibile.
    """
    dataset_dir = get_input_dataset_or_latest(dataset_name)
    df = load_data(os.path.join(dataset_dir, 'metadata.parquet'))
    cluster_centers = compute_cluster_centers(df)

    plot_heatmap(cluster_centers)
    plot_cluster_cardinality(df)
    plot_clusters_per_label(df)
    plot_quartile_counts_by_area(df)
    plot_features_by_quartile(df)

# Permette esecuzione diretta da terminale con argparse
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Genera report grafici per un dataset di clustering.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="Nome della cartella del dataset (es. 'dataset_20240624'). Se omesso, usa il più recente.",
        default=None
    )
    args = parser.parse_args()
    main(dataset_name=args.dataset)

