import random
from enum import Enum
import math
import geojson
import mapbox_vector_tile
import geopy.distance
import requests
import json
from vt2geojson.tools import vt_bytes_to_geojson
import base64
import concurrent.futures
from threading import Lock
import itertools
from typing import Protocol
import os
import utility
from dotenv import load_dotenv
from functools import lru_cache
import logging
from datetime import datetime
import functools
import time

"""
Modulo per il download e l'elaborazione di dataset geospaziali da Mapillary.

Componenti principali:
- Definizione di enumerazioni e interfacce per tipi di feature e strategie di download
- Implementazione di selettori per campionamento dati
- Classe principale Dataminer per gestione configurazioni, download e elaborazione
- Funzioni di utilità per conversione coordinate
"""

# Configurazione del logger: il file di log avrà come nome il timestamp corrente.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"log_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_duration(func):
    """Decorator che logga la durata di esecuzione della funzione decorata."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Operazione {func.__name__} completata in {duration:.2f} secondi")
        return result
    return wrapper

load_dotenv()
API_KEY = os.getenv("MAPILLARY_API_KEY")
GEOJSON_ITALY_PATH = os.getenv("GEOJSON_ITALY_PATH")

session = requests.Session()

# Caching per le chiamate a fetch_features (tile)
@lru_cache(maxsize=128)
def fetch_features_cached(type_call: str, tile_layer: str, x: int, y: int, z: int):
    url = f"https://tiles.mapillary.com/maps/vtp/{type_call}/2/{z}/{x}/{y}?access_token={API_KEY}"
    r = session.get(url)
    r.raise_for_status()
    vt_content = r.content
    return vt_bytes_to_geojson(vt_content, x, y, z, layer=tile_layer)["features"]

# Caching per il recupero della geometria di una immagine (usato in getDistance)
@lru_cache(maxsize=256)
def fetch_image_geometry(image_id: str):
    header = {'Authorization': f'OAuth {API_KEY}'}
    url = f"https://graph.mapillary.com/{image_id}?fields=geometry"
    r = session.get(url, headers=header)
    r.raise_for_status()
    data = r.json()
    return data['geometry']['coordinates']

class Type(Enum):
    ALL = "all"
    LANE_MARKINGS = "marking"
    REGULATORY = "regulatory"
    INFORMATION = "information"
    WARNING = "warning"
    COMPLEMENTARY = "complementary"
    CUSTOM = "custom"

class MapFeatureSelector(Protocol):
    def download_dataset(self, obj, n_threads, geojsonFolder, annotationFolder, imagesFolder, custom_signals, check):
        """Scarica il dataset."""

class BaseSelector:
    def download_dataset(self, obj, n_threads, geojsonFolder, annotationFolder, imagesFolder, custom_signals, check):
        if check == 0:
            # geojsonFolder è il percorso della cartella contenente i GeoJSON
            mapF_id_list = []

            # Ottieni l'elenco di tutti i file GeoJSON nella cartella
            geojson_files = [f for f in os.listdir(geojsonFolder) if f.endswith('.geojson')]

            for geojson_file in geojson_files:
                geojson_path = os.path.join(geojsonFolder, geojson_file)
                # Chiama _select_map_features per ogni file geojson
                mapF_id_list.extend(self._select_map_features(obj, geojson_path))

        elif check == 1:
            mapF_id_list = utility.check_files(geojsonFolder, imagesFolder)

        lock = Lock()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            executor.map(
                obj.process_data,
                mapF_id_list,
                itertools.repeat(lock, len(mapF_id_list)),
                itertools.repeat(imagesFolder, len(mapF_id_list)),
                itertools.repeat(annotationFolder, len(mapF_id_list)),
                itertools.repeat(custom_signals, len(mapF_id_list))
            )

    def _select_map_features(self, obj, geojsonFolder):
        raise NotImplementedError()

class NumberSelector(BaseSelector):
    def _select_map_features(self, obj, geojsonFolder):
        mList = []
        mapF_id_list = []

        try:
            # Gestisci file non trovati o non validi
            with open(geojsonFolder, "r") as file:
                temp = geojson.load(file)
                for feature in temp['features']:
                    mList.append(feature['properties']['id'])

            # Campionamento solo se mList non è vuota. Se è vuota restituisco la lista vuota
            if mList:
                try:
                    mapF_id_list = random.sample(mList, obj.selector['number'])
                except ValueError:
                    logger.info(f"Il file {geojsonFolder} contiene meno feature del numero richiesto. Seleziono tutte le feature disponibili.")
                    mapF_id_list = mList
        except (FileNotFoundError, json.JSONDecodeError):
            logger.error(f"Errore durante l'apertura o la decodifica di {geojsonFolder}.")

        return mapF_id_list

class Dataminer:
    """
    Classe principale per il data mining da Mapillary.
    
    Funzionalità:
    - Configurazione parametri di download
    - Download dataset geospaziali
    - Elaborazione immagini e annotazioni
    - Gestione multi-thread
    """
    def __init__(self):
        self.lines = None
        self.Type = Type.ALL.value
        self.polygons = False
        self.flag = True
        self.dist_min = 10
        self.dist_max = 60
        self.selector = {'percentage': None, 'number': None, 'chunk_dim': None}
        self.counter = 0

    def chooseConfiguration(self, elem):
        """Imposta la configurazione."""
        self.Type = elem.value

    def setPolygon(self, elem):
        """Imposta se utilizzare i poligoni."""
        self.polygons = elem

    def setDistance(self, min=10, max=60):
        """Imposta la distanza minima e massima."""
        self.dist_min = min
        self.dist_max = max

    def setSelector(self, percentage=None, number=None, chunk_dim=None):
        """Imposta i parametri del selettore."""
        self.selector['percentage'] = percentage
        self.selector['number'] = number
        self.selector['chunk_dim'] = chunk_dim

    def getCustomConfiguration(self, configurationFolder):
        """Ottiene la configurazione personalizzata (caricata una sola volta)."""
        if self.lines is None:
            with open(configurationFolder) as file:
                self.lines = [line.rstrip('\n') for line in file]

    def getNGeojson(self, filepath):
        """Ottiene il numero di elementi GeoJSON, gestendo eventuali errori."""
        try:
            with open(filepath, 'r') as f:
                try:
                    temp = geojson.load(f)
                    count = len(temp.get('features', []))
                    return count
                except json.JSONDecodeError:
                    logger.error(f"Errore nel decodificare il file GeoJSON: {filepath}")
                    return 0
        except FileNotFoundError:
            logger.error(f"File GeoJSON non trovato: {filepath}")
            return 0

    def fetch_features(self, type_call, tile_layer, x, y, z):
        """Recupera le feature da Mapillary utilizzando il caching."""
        return fetch_features_cached(type_call, tile_layer, x, y, z)

    @log_duration
    def downloadGeojson(self, ll_lat, ll_lon, ur_lat, ur_lon, z, outputFolder, rows, cols, configurationFolder='empty', output_filename="tsf_data"):
        """
        Scarica dati GeoJSON organizzati in griglia.
        
        Parametri:
        - ll_lat, ll_lon: Lat/lon lower-left bounding box
        - ur_lat, ur_lon: Lat/lon upper-right bounding box
        - z: Livello di zoom
        - outputFolder: Cartella di output
        - rows, cols: Dimensione griglia
        - configurationFolder: Path configurazione personalizzata
        - output_filename: Nome base file output
        
        Funzionalità:
        - Suddivide l'area in celle
        - Limita features per file (max 150)
        - Supporta configurazioni custom
        - Gestisce errori di download
        """
        lat_step = (ur_lat - ll_lat) / rows
        lon_step = (ur_lon - ll_lon) / cols

        for row in range(rows):
            for col in range(cols):
                cell_ll_lat = ll_lat + row * lat_step
                cell_ll_lon = ll_lon + col * lon_step
                cell_ur_lat = cell_ll_lat + lat_step
                cell_ur_lon = cell_ll_lon + lon_step

                cell_llx, cell_lly = deg2num(cell_ll_lat, cell_ll_lon, z)
                cell_urx, cell_ury = deg2num(cell_ur_lat, cell_ur_lon, z)

                output = {"type": "FeatureCollection", "features": []}
                max_features_per_file = 150
                features_count = 0

                try:
                    for x in range(min(cell_llx, cell_urx), max(cell_llx, cell_urx) + 1):
                        for y in range(min(cell_lly, cell_ury), max(cell_lly, cell_ury) + 1):
                            if features_count >= max_features_per_file:
                                break

                            if self.Type == "custom":
                                if configurationFolder == "empty":
                                    raise ValueError("Con la configurazione custom occorre indicare una cartella di configurazione")
                                self.getCustomConfiguration(configurationFolder)
                                features = self.fetch_features("mly_map_feature_traffic_sign", "traffic_sign", x, y, z)
                                for f in features:
                                    for elem in self.lines:
                                        if elem in f['properties']['value']:
                                            output['features'].append(f)
                                            features_count += 1
                            elif self.Type == "all":
                                features = self.fetch_features("mly_map_feature_traffic_sign", "traffic_sign", x, y, z)
                                output['features'].extend(features)
                                features_count += len(features)
                            else:
                                type_call = "mly_map_feature_point" if self.Type == "marking" else "mly_map_feature_traffic_sign"
                                tile_layer = "point" if self.Type == "marking" else "traffic_sign"
                                features = self.fetch_features(type_call, tile_layer, x, y, z)
                                for f in features:
                                    if self.Type in f['properties']['value']:
                                        output['features'].append(f)
                                        features_count += 1

                            if features_count >= max_features_per_file:
                                break
                        if features_count >= max_features_per_file:
                            break

                except requests.exceptions.RequestException as e:
                    logger.error(f"Errore durante il download delle tile: {e}")
                    continue  # Salta alla cella successiva in caso di errore

                output['features'] = output['features'][:max_features_per_file]
                cell_filename = f"{output_filename}_row{row}_col{col}.geojson"
                with open(os.path.join(outputFolder, cell_filename), 'w') as f:
                    json.dump(output, f)

    @log_duration
    def process_data(self, map_feature_id, lock, outputFolderImages, outputFolderAnnotations, custom_signals):
        """
        Elabora una singola feature geospaziale.
        
        Operazioni:
        - Recupero metadati e immagini associate
        - Calcolo geometrie poligonali
        - Verifica presenza segnali personalizzati
        - Download immagine con controllo concorrenza
        - Generazione file annotazione JSON
        - Classificazione posizione geografica
        """
        images_id_list = []
        annotation_data = {
            "map_feature": {},
            "image": {}
        }
        polygon_geometry = None

        header = {'Authorization': f'OAuth {API_KEY}'}
        url = f"https://graph.mapillary.com/{map_feature_id}/detections?fields=image,geometry"
        response = session.get(url, headers=header)
        response.raise_for_status()
        data = response.json()
        for elem in data['data']:
            images_id_list.append(elem['image']['id'])

        if self.polygons:
            polygon_string = data['data'][1]['geometry']
            decoded_data = base64.decodebytes(polygon_string.encode('utf-8'))
            polygon_geometry = mapbox_vector_tile.decode(decoded_data)
            polygon_geometry = polygon_geometry['mpy-or']['features'][0]['geometry']['coordinates']

        url = f"https://graph.mapillary.com/{map_feature_id}?fields=object_value,geometry&access_token={API_KEY}"
        response = session.get(url)
        response.raise_for_status()
        data = response.json()
        annotation_data["map_feature"] = data
        map_feature_coordinates = data['geometry']['coordinates']

        # Controllo: verifico se il punto (in formato [lon, lat]) ricavato ricade in Italia.
        lon, lat = map_feature_coordinates
        if not utility.is_point_in_italy(lon, lat, GEOJSON_ITALY_PATH):
            logger.info("La feature non è in Italia. Interruzione dell'elaborazione.")
            return

        with lock:
            image_id, image_distance = self.getDistance(map_feature_coordinates, images_id_list)
        if image_id is None:
            return
        logger.info(f"Immagine selezionata: {image_id}, Distanza: {image_distance:.2f}m")

        url = f"https://graph.mapillary.com/{image_id}?fields=thumb_original_url, geometry"
        response = session.get(url, headers=header)
        response.raise_for_status()
        data = response.json()
        image_url = data['thumb_original_url']
        annotation_data["image"] = data

        # Controlla se l'immagine contiene i segnali desiderati
        url = f"https://graph.mapillary.com/{image_id}/detections?fields=value,geometry&access_token={API_KEY}"
        response = session.get(url)
        response.raise_for_status()
        detections_data = response.json()["data"]

        download_image = False
        if custom_signals is not None:
            for detection in detections_data:
                if detection['value'] in custom_signals:
                    download_image = True
                    break
        else:
            download_image = True

        if download_image:
            with open(f'{outputFolderImages}/{map_feature_id}.jpg', 'wb') as handler:
                image_data = session.get(image_url, stream=True).content
                handler.write(image_data)

            with open(f'{outputFolderAnnotations}/{map_feature_id}.json', 'w') as handler:
                del annotation_data["image"]['thumb_original_url']
                annotation_data["map_feature"]['value'] = annotation_data["map_feature"].pop('object_value')
                annotation_data["map_feature"]['id_map_feature'] = annotation_data["map_feature"].pop('id')
                annotation_data["map_feature"]['geometry_map_feature'] = annotation_data["map_feature"].pop('geometry')
                annotation_data["image"]['id_image'] = annotation_data["image"].pop('id')
                annotation_data["image"]['distance(m)'] = math.ceil(image_distance)
                annotation_data["image"]['geometry_image'] = annotation_data["image"].pop('geometry')
                if annotation_data["image"]['geometry_image']['coordinates'][1] < 41.5594700:
                    annotation_data["image"]['geographic_location'] = 'sud'
                elif annotation_data["image"]['geometry_image']['coordinates'][1] > 44.801485:
                    annotation_data["image"]['geographic_location'] = 'nord'
                else:
                    annotation_data["image"]['geographic_location'] = 'centre'
                if polygon_geometry:
                    annotation_data["image"]['geometry_polygon'] = polygon_geometry

                annotation_data["image"]['detections'] = detections_data

                for detection in annotation_data["image"]['detections']:
                    base64_string = detection['geometry']
                    decoded_data = base64.decodebytes(base64_string.encode('utf-8'))
                    detection_geometry = mapbox_vector_tile.decode(decoded_data)
                    detection['decoded_geometry'] = detection_geometry

                json.dump(annotation_data, handler)

    @log_duration
    def downloadDataSet(self, n_threads, geojsonFolder, annotationFolder, imagesFolder, download_strategy: MapFeatureSelector, custom_signals=None, check=-1):
        """Scarica il dataset utilizzando la strategia specificata."""
        download_strategy.download_dataset(self, n_threads, geojsonFolder, annotationFolder, imagesFolder, custom_signals, check)

    def getDistance(self, coordinatesMapF, imageIdList):
        """Calcola la distanza tra le coordinate della Map Feature e le immagini."""
        min_distance = self.dist_max + 1
        selected_image_id = None
        for image_id in imageIdList:
            coordinatesImageId = fetch_image_geometry(image_id)
            currDist = geopy.distance.geodesic(coordinatesMapF, coordinatesImageId).m
            if self.dist_min <= currDist < min_distance:
                min_distance = currDist
                selected_image_id = image_id
        return selected_image_id, min_distance

def deg2num(lat_deg, lon_deg, zoom):
    """
    Converte coordinate geografiche a coordinate tile XYZ.
    
    Parametri:
    - lat_deg: Latitudine decimale
    - lon_deg: Longitudine decimale
    - zoom: Livello di zoom
    
    Restituisce:
    - (x, y): Coordinate tile
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x_tile = int((lon_deg + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile