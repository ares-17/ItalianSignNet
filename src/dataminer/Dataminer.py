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
import functools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
API_KEY = os.getenv("MAPILLARY_API_KEY")
GEOJSON_ITALY_PATH = os.getenv("GEOJSON_ITALY_PATH")
GRID_WORKERS = int(os.getenv("DATAMINER_GRID_WORKERS", 5))

session = requests.Session()

def log_duration(func):
    """Decorator che logga la durata di esecuzione della funzione decorata.
       Se il primo argomento (self) possiede l'attributo 'logger', lo utilizza."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        # Se la funzione è un metodo d'istanza e self.logger esiste, lo usa:
        instance_logger = args[0].logger if args and hasattr(args[0], 'logger') else None
        if instance_logger:
            instance_logger.info(f"Operazione {func.__name__} completata in {duration:.2f} secondi")
        else:
            print(f"Operazione {func.__name__} completata in {duration:.2f} secondi")
        return result
    return wrapper

# Caching per le chiamate a fetch_features (tile)
@lru_cache(maxsize=128)
def fetch_features_cached(type_call: str, tile_layer: str, x: int, y: int, z: int):
    url = f"https://tiles.mapillary.com/maps/vtp/{type_call}/2/{z}/{x}/{y}?access_token={API_KEY}"
    max_attempts = 3  # numero massimo di tentativi
    for attempt in range(max_attempts):
        try:
            r = session.get(url)
            r.raise_for_status()
            vt_content = r.content
            geojson_data = vt_bytes_to_geojson(vt_content, x, y, z, layer=tile_layer)
            return geojson_data["features"]
        except Exception as e:
            # Non utilizziamo qui il logger globale, poiché il decorator e le istanze dovrebbero gestire il logging.
            print(f"Errore nel decodificare il vector tile per x:{x}, y:{y}, z:{z} - {e} (tentativo {attempt+1}/{max_attempts})")
            time.sleep(0.5)
    return []

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
                    obj.logger.info(f"Il file {geojsonFolder} contiene meno feature del numero richiesto. Seleziono tutte le feature disponibili.")
                    mapF_id_list = mList
        except (FileNotFoundError, json.JSONDecodeError):
            obj.logger.error(f"Errore durante l'apertura o la decodifica di {geojsonFolder}.")

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
    def __init__(self, logger):
        self.logger = logger
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
        self.logger.info(f"Configurazione scelta: {self.Type}")

    def setPolygon(self, elem):
        """Imposta se utilizzare i poligoni."""
        self.polygons = elem
        self.logger.info(f"Impostato polygons: {self.polygons}")

    def setDistance(self, min=10, max=60):
        """Imposta la distanza minima e massima."""
        self.dist_min = min
        self.dist_max = max
        self.logger.info(f"Impostata distanza: min={self.dist_min}, max={self.dist_max}")

    def setSelector(self, percentage=None, number=None, chunk_dim=None):
        """Imposta i parametri del selettore."""
        self.selector['percentage'] = percentage
        self.selector['number'] = number
        self.selector['chunk_dim'] = chunk_dim
        self.logger.info(f"Impostato selector: {self.selector}")

    def getCustomConfiguration(self, configurationFolder):
        """Ottiene la configurazione personalizzata (caricata una sola volta)."""
        if self.lines is None:
            with open(configurationFolder) as file:
                self.lines = [line.rstrip('\n') for line in file]
            self.logger.info(f"Caricata configurazione custom da {configurationFolder}")

    def getNGeojson(self, filepath):
        """Ottiene il numero di elementi GeoJSON, gestendo eventuali errori."""
        try:
            with open(filepath, 'r') as f:
                try:
                    temp = geojson.load(f)
                    count = len(temp.get('features', []))
                    return count
                except json.JSONDecodeError:
                    self.logger.error(f"Errore nel decodificare il file GeoJSON: {filepath}")
                    return 0
        except FileNotFoundError:
            self.logger.error(f"File GeoJSON non trovato: {filepath}")
            return 0

    def fetch_features(self, type_call, tile_layer, x, y, z):
        """Recupera le feature da Mapillary utilizzando il caching."""
        return fetch_features_cached(type_call, tile_layer, x, y, z)

    @log_duration
    def downloadGeojson(self, ll_lat, ll_lon, ur_lat, ur_lon, z, outputFolder, rows, cols, configurationFolder='empty', output_filename="tsf_data"):
        """
        Scarica dati GeoJSON organizzati in griglia in parallelo per ogni cella.
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

                tile_coords = []
                for x in range(min(cell_llx, cell_urx), max(cell_llx, cell_urx) + 1):
                    for y in range(min(cell_lly, cell_ury), max(cell_lly, cell_ury) + 1):
                        tile_coords.append((x, y))

                features_count = 0

                # Usa un ThreadPoolExecutor per fare richieste in parallelo
                with ThreadPoolExecutor(max_workers=GRID_WORKERS) as executor:
                    future_to_tile = {
                        executor.submit(self.fetch_features, 
                                        "mly_map_feature_traffic_sign" if self.Type in ["all", "custom"] else (
                                            "mly_map_feature_point" if self.Type == "marking" else "mly_map_feature_traffic_sign"
                                        ),
                                        "traffic_sign" if self.Type not in ["marking"] else "point",
                                        x, y, z
                        ): (x, y) for x, y in tile_coords
                    }
                    for future in as_completed(future_to_tile):
                        x, y = future_to_tile[future]
                        try:
                            features = future.result()
                        except Exception as e:
                            self.logger.error(f"Errore nel fetch del tile x:{x}, y:{y} - {e}")
                            features = []
                            
                        # Applica la logica per selezionare le feature in base a self.Type
                        if self.Type == "custom":
                            if configurationFolder == "empty":
                                raise ValueError("Con la configurazione custom occorre indicare una cartella di configurazione")
                            self.getCustomConfiguration(configurationFolder)
                            for f in features:
                                for elem in self.lines:
                                    if elem in f['properties']['value']:
                                        output['features'].append(f)
                                        features_count += 1
                                        if features_count >= max_features_per_file:
                                            break
                        elif self.Type == "all":
                            output['features'].extend(features)
                            features_count += len(features)
                        else:
                            for f in features:
                                if self.Type in f['properties']['value']:
                                    output['features'].append(f)
                                    features_count += 1
                                    if features_count >= max_features_per_file:
                                        break
                                    
                        # esci dal loop se abbiamo raggiunto il massimo
                        if features_count >= max_features_per_file:
                            break
                        
                output['features'] = output['features'][:max_features_per_file]
                cell_filename = f"{output_filename}_row{row}_col{col}.geojson"
                with open(os.path.join(outputFolder, cell_filename), 'w') as f:
                    json.dump(output, f)

    @log_duration
    def process_data(self, map_feature_id, lock, outputFolderImages, outputFolderAnnotations, custom_signals):
        """
        Elabora una singola feature geospaziale.
        """
        images_id_list = []
        annotation_data = {"map_feature": {}, "image": {}}
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

        # Verifica se la feature è in Italia
        lon, lat = map_feature_coordinates
        if not utility.is_point_in_italy(lon, lat, GEOJSON_ITALY_PATH):
            self.logger.info("La feature non è in Italia. Interruzione dell'elaborazione.")
            return

        with lock:
            image_id, image_distance = self.getDistance(map_feature_coordinates, images_id_list)
        if image_id is None:
            return
        self.logger.info(f"Immagine selezionata: {image_id}, Distanza: {image_distance:.2f}m")

        url = f"https://graph.mapillary.com/{image_id}?fields=thumb_original_url, geometry"
        response = session.get(url, headers=header)
        response.raise_for_status()
        data = response.json()
        image_url = data['thumb_original_url']
        annotation_data["image"] = data

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
    """
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x_tile = int((lon_deg + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x_tile, y_tile