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
import logging
from typing import Optional, Tuple
import sys

utils_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(utils_path)

from utils.MunicipalGeocoder import MunicipalGeocoder

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
DATA_SOURCE_ROOT = os.path.join(BASE_DIR or '', os.getenv("TEST_CASE_BASE_ROOT") or '')
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
OUTPUT_DIR = os.path.join(BASE_DIR or '', 'src', 'dataset', 'artifacts', f'dataset_{TIMESTAMP}_eps_{DBSCAN_DISTANCE}')
LABEL_INDEX_FILE = os.path.join(BASE_DIR or '', 'src', 'utils', 'signnames.csv')
MUNICIPALITIES_LIMIT_IT = os.path.join(BASE_DIR or '', 'src', 'resources', 'limits_IT_municipalities.geojson')
REGIONS_LIMIT_IT = os.path.join(BASE_DIR or '', 'src', 'resources', 'limits_IT_regions.geojson')
REDDITI_IRPEF_2023_IT = os.path.join(BASE_DIR or '', 'src', 'resources', 'Redditi_e_principali_variabili_IRPEF_su_base_subcomunale_CSV_2023.csv')

mlflow.set_tracking_uri('http://localhost:5000')

def get_latest_json() -> str:
    logs_dir = os.path.join(BASE_DIR or '', 'src', 'dataset', 'logs')
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


def parse_coordinates(coord_string: str) -> Optional[Tuple[float, float]]:
    """
    Parse coordinate string in format "latitude,longitude" to tuple of floats.
    
    Args:
        coord_string: String in format "lat,lon" (e.g., "45.95898155906983,12.626659870147705")
        
    Returns:
        Tuple of (latitude, longitude) or None if parsing fails
    """
    try:
        if pd.isna(coord_string) or not isinstance(coord_string, str):
            return None
        
        # Remove any whitespace and split by comma
        coords = coord_string.strip().split(',')
        if len(coords) != 2:
            return None

        return float(coords[0]), float(coords[1])
        
    except (ValueError, AttributeError) as e:
        logging.warning(f"Failed to parse coordinates '{coord_string}': {e}")
        return None

def add_municipality_codes_to_dataframe(
    df: pd.DataFrame, 
    regions_file: str, 
    municipalities_file: str,
    coordinates_column: str = 'coordinates',
    output_column: str = 'com_istat_code',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Add municipality ISTAT codes to a DataFrame based on coordinates.
    
    Args:
        df: Input DataFrame with coordinates
        regions_file: Path to regions GeoJSON file
        municipalities_file: Path to municipalities GeoJSON file
        coordinates_column: Name of the column containing coordinates (default: 'coordinates')
        output_column: Name of the output column for ISTAT codes (default: 'com_istat_code')
        logger: Optional logger instance
        
    Returns:
        DataFrame with added municipality codes column
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Initialize the geocoder
    logger.info("Initializing municipal geocoder...")
    geocoder = MunicipalGeocoder(regions_file, municipalities_file, logger)
    
    # Check if coordinates column exists
    if coordinates_column not in df.columns:
        raise ValueError(f"Column '{coordinates_column}' not found in DataFrame")
    
    # Parse coordinates and geocode
    logger.info(f"Processing {len(df)} rows...")
    municipality_codes = []
    successful_geocodings = 0
    
    for idx, coord_string in enumerate(df[coordinates_column]):
        if idx % 100 == 0 and idx > 0:
            logger.info(f"Processed {idx}/{len(df)} rows...")
        
        # Parse coordinates
        coords = parse_coordinates(coord_string)
        if coords is None:
            municipality_codes.append(None)
            continue
        
        lat, lon = coords
        
        # Geocode
        result = geocoder.geocode(lat, lon)
        if result:
            municipality_codes.append(result['com_istat_code'])
            successful_geocodings += 1
        else:
            municipality_codes.append(None)
    
    result_df[output_column] = municipality_codes
    logger.info(f"Geocoding completed: {successful_geocodings}/{len(df)} coordinates successfully geocoded")
    
    return result_df

def add_municipality_codes_batch(
    df: pd.DataFrame, 
    regions_file: str, 
    municipalities_file: str,
    coordinates_column: str = 'coordinates',
    output_column: str = 'com_istat_code',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Add municipality ISTAT codes to a DataFrame using batch processing for better performance.
    
    Args:
        df: Input DataFrame with coordinates
        regions_file: Path to regions GeoJSON file
        municipalities_file: Path to municipalities GeoJSON file
        coordinates_column: Name of the column containing coordinates (default: 'coordinates')
        output_column: Name of the output column for ISTAT codes (default: 'com_istat_code')
        logger: Optional logger instance
        
    Returns:
        DataFrame with added municipality codes column
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Initialize the geocoder
    logger.info("Initializing municipal geocoder...")
    geocoder = MunicipalGeocoder(regions_file, municipalities_file, logger)
    
    # Check if coordinates column exists
    if coordinates_column not in df.columns:
        raise ValueError(f"Column '{coordinates_column}' not found in DataFrame")
    
    # Parse all coordinates
    logger.info("Parsing coordinates...")
    coordinate_tuples = []
    valid_indices = []
    
    for idx, coord_string in enumerate(df[coordinates_column]):
        coords = parse_coordinates(coord_string)
        if coords is not None:
            coordinate_tuples.append(coords)
            valid_indices.append(idx)
    
    logger.info(f"Found {len(coordinate_tuples)} valid coordinates out of {len(df)} rows")
    
    # Batch geocode
    logger.info("Performing batch geocoding...")
    geocoding_results = geocoder.geocode_batch(coordinate_tuples)
    
    # Create result array
    municipality_codes = [None] * len(df)
    successful_geocodings = 0
    
    for i, result in enumerate(geocoding_results):
        original_idx = valid_indices[i]
        if result:
            municipality_codes[original_idx] = result['com_istat_code']
            successful_geocodings += 1
    
    # Add the new column
    result_df[output_column] = municipality_codes
    
    logger.info(f"Batch geocoding completed: {successful_geocodings}/{len(coordinate_tuples)} coordinates successfully geocoded")
    
    return result_df

def define_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"create_dataset_{timestamp_str}_eps_{DBSCAN_DISTANCE}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_filename)]
    )
    return logging.getLogger(__name__)

def add_income_to_metadata(metadata_df, income_csv_path, logger: logging.Logger):
    """
    Aggiunge la colonna del reddito complessivo per comune al dataframe metadata.
    
    Args:
        metadata_df: DataFrame con le colonne filename, feature, feature_index, coordinates, com_istat_code
        income_csv_path: Path del file CSV contenente i dati del reddito
    
    Returns:
        DataFrame con la nuova colonna 'total_income_per_municipality'
    """
    
    try:
        income_df = pd.read_csv(income_csv_path, sep=';', encoding='utf-8')
        logger.info(f"File CSV caricato con successo. Shape: {income_df.shape}")
    except Exception as e:
        logger.error(f"Errore nella lettura del file CSV: {e}")
        return metadata_df
    
    istat_code_col = None
    income_amount_col = None
    
    for col in income_df.columns:
        col_lower = col.lower().strip()
        if 'codice istat comune' in col_lower:
            istat_code_col = col
        elif 'reddito complessivo - ammontare in euro' in col_lower:
            income_amount_col = col
    
    if istat_code_col is None:
        logger.error("ERRORE: Colonna 'Codice Istat Comune' non trovata nel CSV")
        return metadata_df
    
    if income_amount_col is None:
        logger.error("ERRORE: Colonna 'Reddito complessivo - Ammontare in euro' non trovata nel CSV")
        return metadata_df
    
    logger.info(f"Colonna codice ISTAT trovata: '{istat_code_col}'")
    logger.info(f"Colonna reddito trovata: '{income_amount_col}'")
    
    # Seleziona solo le colonne necessarie
    income_subset = income_df[[istat_code_col, income_amount_col]].copy()
    
    # Converti i codici ISTAT in string di 6 caratteri
    income_subset[istat_code_col] = income_subset[istat_code_col].astype(str).str.zfill(6)
    
    # Gestisci i valori mancanti o non numerici nella colonna del reddito
    income_subset[income_amount_col] = pd.to_numeric(income_subset[income_amount_col], errors='coerce')
    
    # Rimuovi le righe con valori mancanti
    income_subset = income_subset.dropna()
    
    logger.info(f"Righe valide per l'aggregazione: {len(income_subset)}")
    
    # Aggrega i dati per codice ISTAT (somma i redditi per comune)
    logger.info("Aggregazione dei dati per codice ISTAT...")
    income_aggregated = income_subset.groupby(istat_code_col)[income_amount_col].sum().reset_index()
    income_aggregated.columns = ['com_istat_code', 'total_income_per_municipality']
    
    # Converti anche la colonna com_istat_code del metadata in string
    metadata_df = metadata_df.copy()
    metadata_df['com_istat_code'] = metadata_df['com_istat_code'].astype(str)
    
    logger.info(f"Comuni unici nel dataset del reddito: {len(income_aggregated)}")
    logger.info(f"Comuni unici nel metadata: {len(metadata_df['com_istat_code'].unique())}")
    
    # Merge dei dati
    print("Unione dei dati...")
    result_df = metadata_df.merge(
        income_aggregated, 
        on='com_istat_code', 
        how='left'
    )
    
    # Verifica quanti match sono stati trovati
    matched_rows = result_df['total_income_per_municipality'].notna().sum()
    total_rows = len(result_df)
    
    logger.info(f"Match trovati: {matched_rows}/{total_rows} righe ({matched_rows/total_rows*100:.1f}%)")
    
    # Mostra alcuni esempi di codici ISTAT che non hanno match
    unmatched_codes = result_df[result_df['total_income_per_municipality'].isna()]['com_istat_code'].unique()
    if len(unmatched_codes) > 0:
        logger.warning(f"Esempi di codici ISTAT senza match: {unmatched_codes[:5]}")
    
    # Mostra statistiche della nuova colonna
    if matched_rows > 0:
        logger.info(f"\nStatistiche della colonna 'total_income_per_municipality':")
        logger.info(f"Media: €{result_df['total_income_per_municipality'].mean():,.0f}")
        logger.info(f"Mediana: €{result_df['total_income_per_municipality'].median():,.0f}")
        logger.info(f"Min: €{result_df['total_income_per_municipality'].min():,.0f}")
        logger.info(f"Max: €{result_df['total_income_per_municipality'].max():,.0f}")
    
    return result_df

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = define_logger()
    
    with open(CLUSTER_JSON_PATH) as f:
        clustersInfo: dict = json.load(f)
    
    create_folder_sets()

    clusters: dict = clustersInfo['clusters_by_label']
    metadata: pd.DataFrame = pd.read_csv(os.path.join(DATA_SOURCE_ROOT, "annotations.csv"))
    metadata = df_add_label_index_column(metadata)
    metadata = add_coordinates_to_dataframe(metadata, clustersInfo['coordinates'])
    metadata = add_municipality_codes_batch(
        df=metadata,
        regions_file=REGIONS_LIMIT_IT,
        municipalities_file=MUNICIPALITIES_LIMIT_IT,
        logger=logger
    )
    metadata = add_income_to_metadata(metadata, REDDITI_IRPEF_2023_IT, logger)
    metadata = assign_cluster_id(metadata, clusters)
    metadata = split_dataset(metadata)
    pd.set_option('display.max_columns', None)
    print(metadata.head())
    return
    metadata = copy_images_to_output_path(metadata)

    save_to_parquet(metadata)
    log_dataset_info(metadata, clustersInfo['report'])

if __name__ == "__main__":
    main()