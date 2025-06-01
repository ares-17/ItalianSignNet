from Dataminer import Dataminer, Type, NumberSelector
import utility
import os
from dotenv import load_dotenv
from pathlib import Path
import logging
from datetime import datetime
import argparse

"""
Script per il download e processing di dati geospaziali dall'API Mapillary.

Flusso principale:
1. Configurazione parametri di estrazione
2. Gestione cartelle e percorsi
3. Download dati grezzi in formato GeoJSON
4. Selezione features e download immagini/annotazioni
5. Post-processing con utilità esterne

Dipendenza principale: modulo Dataminer.py
"""

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR", ""))
NO_ASK_TO_OVWEWRITE_OLD_TESTS = os.getenv("NO_ASK_TO_OVWEWRITE_OLD_TESTS", False)
n_threads = 4
cartellaBase = os.path.join(BASE_DIR / 'testing')
z = 14  # Livello di zoom per le tile

parser = argparse.ArgumentParser(
    description="Script per il download e processing di dati geospaziali dall'API Mapillary."
)
parser.add_argument("--ll_lat", type=float, default=41.902277, help="Latitudine inferiore (Lower Left)")
parser.add_argument("--ll_lon", type=float, default=12.250977, help="Longitudine inferiore (Lower Left)")
parser.add_argument("--ur_lat", type=float, default=43.897892, help="Latitudine superiore (Upper Right)")
parser.add_argument("--ur_lon", type=float, default=14.458008, help="Longitudine superiore (Upper Right)")
parser.add_argument("--num_features", type=int, default=10, help="Numero di feature da estrarre da ogni file GeoJSON")
parser.add_argument("--nome_esecuzione", type=str, default="centro", help="Nome dell'esecuzione")
args = parser.parse_args()

ll_lat = args.ll_lat
ll_lon = args.ll_lon
ur_lat = args.ur_lat
ur_lon = args.ur_lon
nome_esecuzione = args.nome_esecuzione

def setup_logger(level=logging.INFO) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cartellaBase, nome_esecuzione , f"{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Configurazione percorsi di output
geojsonFolder = os.path.join(cartellaBase, nome_esecuzione, "geojson_folder")
outputFolderImages = os.path.join(cartellaBase, nome_esecuzione, "images")
outputFolderAnnotationsImage = os.path.join(cartellaBase, nome_esecuzione, "annotations_image")
outputFolderBounded = os.path.join(cartellaBase, nome_esecuzione, "bounded_images")
outputFolderCSV = os.path.join(cartellaBase, nome_esecuzione, "signal_catalog")
outputRitagli = os.path.join(cartellaBase, nome_esecuzione, "resized_images")

percorso_esecuzione = os.path.join(cartellaBase, nome_esecuzione)
geojson_file_path = os.path.join(geojsonFolder, "mapF_id_list.txt")

percorso_configurazione = BASE_DIR / 'src/resources/traffic-signs.txt'

with open(percorso_configurazione, 'r') as f:
    custom_signals = [line.strip() for line in f]

#Impostazione righe e colonne in cui suddividere l'area
rows = 6
cols = 12


if os.path.exists(percorso_esecuzione):
    logger = setup_logger()
    data = Dataminer(logger)
    data.chooseConfiguration(Type.CUSTOM)

    if os.path.exists(geojson_file_path) and not os.listdir(outputRitagli):
        continua_download = input("Esecuzione già iniziata, vuoi continuare il download? (s/n): ")
        if continua_download.lower() == 's':
            data.downloadDataSet(n_threads, geojson_file_path, outputFolderAnnotationsImage, outputFolderImages,
                                 NumberSelector(), custom_signals=custom_signals, check=1)
        elif continua_download.lower() == 'n':
            sovrascrivi = input("Vuoi sovrascrivere l'esecuzione esistente? (s/n): ")
            if sovrascrivi.lower() == 's':
                for cartella in [outputFolderImages, outputFolderAnnotationsImage, outputFolderBounded, outputFolderCSV,
                                 outputRitagli]:
                    utility.clear_folder(cartella)

                percorso_mappa = os.path.join(cartellaBase, nome_esecuzione, f"{nome_esecuzione}_immagini.html")
                if os.path.exists(percorso_mappa):
                    os.remove(percorso_mappa)

                geojsonFilePathList = []  # Crea la lista di percorsi GeoJSON
                for row in range(rows):
                    for col in range(cols):
                        geojsonFilePathList.append(os.path.join(geojsonFolder,
                                                                f"{nome_esecuzione}_row{row}_col{col}.geojson"))

                mapF_id_list = []  # Crea la lista di ID delle feature
                for geojsonFilePath in geojsonFilePathList:  # Itera sui file GeoJSON
                    n = data.getNGeojson(geojsonFilePath)
                    if n > 0:
                        data.setSelector(number=min(args.num_features, n))
                        data.setDistance(min=11, max=40) # Imposta distanza min e max
                        data.setPolygon(True)

                        number_selector = NumberSelector()
                        selected_features = number_selector._select_map_features(data, geojsonFilePath)

                        if selected_features:
                            mapF_id_list.extend(selected_features)   # Aggiungi le features selezionate alla lista

                id_list_filepath = os.path.join(geojsonFolder, "mapF_id_list.txt")  #Crea il percorso del file di testo
                with open(id_list_filepath, "w") as f:   # Apri il file in scrittura
                    for id_val in mapF_id_list: # Scrivi gli ID delle feature nel file
                        f.write(str(id_val) + "\n")

                data.downloadDataSet(n_threads, geojsonFolder, outputFolderAnnotationsImage, outputFolderImages,
                                     NumberSelector(), custom_signals=custom_signals, check=0)  # Avvia il download
            elif sovrascrivi.lower() == 'n':
                print("Esecuzione annullata.")
                exit()
            else:
                print("Input non valido. Inserisci 's' o 'n'.")
                exit()
        else:
            print("Input non valido. Inserisci 's' o 'n'.")
            exit()

    elif os.listdir(outputRitagli):  # Esecuzione già completata - qui va lo stesso codice della sovrascrittura
        if NO_ASK_TO_OVWEWRITE_OLD_TESTS:
            sovrascrivi = 'n'
        else:
            sovrascrivi = input("Esecuzione già esistente. Vuoi sovrascriverla? (s/n): ")
        
        if sovrascrivi.lower() == 's':
            for cartella in [outputFolderImages, outputFolderAnnotationsImage, outputFolderBounded, outputFolderCSV,
                             outputRitagli]:
                utility.clear_folder(cartella)

            geojsonFilePathList = []  # Crea la lista di percorsi GeoJSON
            for row in range(rows):
                for col in range(cols):
                    geojsonFilePathList.append(os.path.join(geojsonFolder, f"{nome_esecuzione}_row{row}_col{col}.geojson"))

            mapF_id_list = []  # Crea la lista di ID delle feature

            for geojsonFilePath in geojsonFilePathList:  # Itera sui file GeoJSON
                n = data.getNGeojson(geojsonFilePath)
                if n > 0:
                    data.setSelector(number=min(args.num_features, n))  # Imposta il numero di feature da estrarre
                    data.setDistance(min=11, max=40)  # Imposta distanza min e max
                    data.setPolygon(True)

                    number_selector = NumberSelector()
                    selected_features = number_selector._select_map_features(data,
                                                                               geojsonFilePath)  # Seleziona le features

                    if selected_features:
                        mapF_id_list.extend(selected_features)  # Aggiungi le features selezionate alla lista

            id_list_filepath = os.path.join(geojsonFolder, "mapF_id_list.txt")  # Crea il percorso del file di testo
            with open(id_list_filepath, "w") as f:  # Apri il file in scrittura
                for id_val in mapF_id_list:  # Scrivi gli ID delle feature nel file
                    f.write(str(id_val) + "\n")

            data.downloadDataSet(n_threads, geojsonFolder, outputFolderAnnotationsImage, outputFolderImages,
                                 NumberSelector(), custom_signals=custom_signals, check=0)  # Avvia il download
        elif sovrascrivi.lower() == 'n':
            print("Esecuzione annullata.")
            exit()
        else:
            print("Input non valido. Inserisci 's' o 'n'.")
            exit()
else:  # Nuova esecuzione
    utility.folder_maker(cartellaBase, nome_esecuzione)
    logger = setup_logger()
    data = Dataminer(logger)
    data.chooseConfiguration(Type.CUSTOM)

    data.downloadGeojson(ll_lat, ll_lon, ur_lat, ur_lon, z, geojsonFolder, rows, cols, percorso_configurazione,
                         nome_esecuzione)
    geojsonFilePathList = []  # Crea la lista di percorsi GeoJSON
    for row in range(rows):
        for col in range(cols):
            geojsonFilePathList.append(os.path.join(geojsonFolder, f"{nome_esecuzione}_row{row}_col{col}.geojson"))

    mapF_id_list = []

    for geojsonFilePath in geojsonFilePathList:  # Itera sui file GeoJSON
        n = data.getNGeojson(geojsonFilePath)
        if n > 0:
            data.setSelector(number=min(args.num_features, n))  # Imposta il numero di feature da estrarre
            data.setDistance(min=11, max=40)  # Imposta distanza min e max
            data.setPolygon(True)

            number_selector = NumberSelector()
            selected_features = number_selector._select_map_features(data, geojsonFilePath)

            if selected_features:
                mapF_id_list.extend(selected_features)  # Aggiungi le features selezionate alla lista

    id_list_filepath = os.path.join(geojsonFolder, "mapF_id_list.txt")  # Crea il percorso del file di testo
    with open(id_list_filepath, "w") as f:  # Apri il file in scrittura
        for id_val in mapF_id_list:  # Scrivi gli ID delle feature nel file
            f.write(str(id_val) + "\n")

    data.downloadDataSet(n_threads, geojsonFolder, outputFolderAnnotationsImage, outputFolderImages,
                         NumberSelector(), custom_signals=custom_signals, check=0)  # Avvia il download


utility.bounding(outputFolderImages, outputFolderAnnotationsImage, outputFolderBounded)
utility.csv_maker(outputFolderAnnotationsImage, outputFolderCSV, nome_esecuzione)
utility.resizer(outputFolderImages, outputFolderAnnotationsImage, outputRitagli, custom_signals, percorso_esecuzione)
utility.map_maker(outputFolderAnnotationsImage, ll_lat, ll_lon, ur_lat, ur_lon, nome_esecuzione, cartellaBase)

# Libera spazio occupato da cartelle con immagini originali. 
# Rimovere le seguenti righe se le immagini originali sono necessarie
utility.safe_clear_folder(outputFolderBounded)
utility.safe_clear_folder(outputFolderImages)