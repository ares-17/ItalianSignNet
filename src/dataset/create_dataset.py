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
from utils.RegionGeocoder import RegionGeocoder

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
DATA_SOURCE_ROOT = os.path.join(BASE_DIR or '', os.getenv("TEST_CASE_BASE_ROOT") or '')
DBSCAN_DISTANCE = int(os.getenv("DBSCAN_DISTANCE", 100))
OUTPUT_DIR = os.path.join(BASE_DIR or '', 'src', 'dataset', 'artifacts', f'dataset_{TIMESTAMP}_eps_{DBSCAN_DISTANCE}')

# files
LABEL_INDEX_FILE = os.path.join(BASE_DIR or '', 'src', 'utils', 'signnames.csv')
MUNICIPALITIES_LIMIT_IT = os.path.join(BASE_DIR or '', 'src', 'resources', 'limits_IT_municipalities.geojson')
REGIONS_LIMIT_IT = os.path.join(BASE_DIR or '', 'src', 'resources', 'limits_IT_regions.geojson')
REDDITI_IRPEF_IT = os.path.join(BASE_DIR or '', 'src', 'resources', 'Redditi_e_principali_variabili_IRPEF_su_base_comunale_CSV_2023.csv')

# IRPEF INCOME COLUMNS
ISTAT_CODE_COLUMN_METADATA = 'com_istat_code'
ISTAT_CODE_COLUMN_CSV = 'Codice Istat Comune'
INCOME_COLUMN_CSV = 'Reddito complessivo - Ammontare in euro'
COUNT_PER_INCOME = 'Numero contribuenti'
COLUMN_AVERAGE_INCOME = 'average_income'

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

    metadata.loc[:, 'split'] = 'train'
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

def log_dataset_info(metadata: pd.DataFrame, cluster_report: dict, cluster_json_path: str) -> None:
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
            "json_spatial_clustering": cluster_json_path
        })

        mlflow.log_artifact(cluster_json_path)

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

def add_income_column(df, logger, csv_file_path):
    logger.info("Starting process to add total income column")

    try:
        income_df = pd.read_csv(csv_file_path, sep=';')
        logger.info(f"Loaded {len(income_df)} rows from income CSV")

        df[ISTAT_CODE_COLUMN_METADATA] = df[ISTAT_CODE_COLUMN_METADATA].astype(str).str.zfill(6)
        income_df[ISTAT_CODE_COLUMN_CSV] = income_df[ISTAT_CODE_COLUMN_CSV].astype(str).str.zfill(6)

        income_df['Reddito medio'] = income_df[INCOME_COLUMN_CSV] / income_df[COUNT_PER_INCOME]
        
        income_mapping = dict(zip(income_df[ISTAT_CODE_COLUMN_CSV], income_df['Reddito medio']))
        df[COLUMN_AVERAGE_INCOME] = df[ISTAT_CODE_COLUMN_METADATA].map(income_mapping)

        df['income_quartile'] = pd.qcut(df[COLUMN_AVERAGE_INCOME], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

        logger.info("Successfully added income column and quartile classification.")
        return df

    except Exception as e:
        logger.error(f"Error while adding income column: {str(e)}")
        raise

def add_macro_area_column(
    df: pd.DataFrame, 
    regions_file: str,
    coordinates_column: str = 'coordinates',
    output_column: str = 'area',
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Aggiunge una colonna 'area' al DataFrame con il valore 'nord', 'centre' o 'sud',
    calcolato a partire dalle coordinate di ciascuna riga usando RegionGeocoder.
    
    Args:
        df: DataFrame in ingresso contenente una colonna di coordinate
        regions_file: Path al file GeoJSON delle regioni italiane
        coordinates_column: Nome della colonna con le coordinate (formato stringa)
        output_column: Nome della colonna di output da creare (default: 'area')
        logger: Logger opzionale

    Returns:
        DataFrame con la colonna 'area' aggiunta
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Inizializzazione RegionGeocoder...")
    geocoder = RegionGeocoder(regions_file, logger)

    if coordinates_column not in df.columns:
        raise ValueError(f"La colonna '{coordinates_column}' non è presente nel DataFrame.")
    
    # Prepara coordinate valide
    coordinate_tuples = []
    valid_indices = []
    logger.info("Parsing coordinate valide...")
    for idx, coord_string in enumerate(df[coordinates_column]):
        coords = parse_coordinates(coord_string)
        if coords is not None:
            coordinate_tuples.append(coords)
            valid_indices.append(idx)

    logger.info(f"Trovate {len(coordinate_tuples)} coordinate valide su {len(df)} righe.")

    # Batch geocoding
    logger.info("Esecuzione batch geocoding...")
    geocoding_results = geocoder.geocode_batch(coordinate_tuples)

    # Inizializza la nuova colonna
    macro_areas = [None] * len(df)
    successful = 0

    for i, result in enumerate(geocoding_results):
        original_idx = valid_indices[i]
        if result and result.get("macro_area"):
            macro_areas[original_idx] = result["macro_area"]
            successful += 1

    logger.info(f"Macro-area assegnata con successo a {successful}/{len(coordinate_tuples)} righe.")

    result_df = df.copy()
    result_df[output_column] = macro_areas

    return result_df

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger = define_logger()
    
    cluster_json_path = get_latest_json()

    with open(cluster_json_path) as f:
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
    metadata = add_income_column(
        df=metadata,
        csv_file_path=REDDITI_IRPEF_IT,
        logger=logger
    )
    metadata = assign_cluster_id(metadata, clusters)
    metadata = split_dataset(metadata)
    metadata = add_macro_area_column(
        df=metadata,
        regions_file=REGIONS_LIMIT_IT,
        coordinates_column='coordinates',
        logger=logger,
        output_column='area',
    )

    pd.set_option('display.max_columns', None)
    print(metadata.head())
    #metadata = copy_images_to_output_path(metadata)
    save_to_parquet(metadata)
    log_dataset_info(metadata, clustersInfo['report'], cluster_json_path)

if __name__ == "__main__":
    main()