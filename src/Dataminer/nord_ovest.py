from Dataminer import Dataminer, Type, NumberSelector
import utility
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from regioni import REGIONI

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
BASE_DIR = Path(os.getenv("BASE_DIR"))

n_threads = 4

cartellaBase = BASE_DIR / 'testing' #Inserire percorso locale per testing
nome_esecuzione = "NordOvest_data" #Inserire nome cartella esecuzione da generare

# Coordinate area di interesse (bounding box)
z = 14  # Livello di zoom per le tile
ll_lat = 44.071020  # Latitudine angolo inferiore sinistro
ll_lon = 7.754966   # Longitudine angolo inferiore sinistro
ur_lat = 46.167202  # Latitudine angolo superiore destro
ur_lon = 9.991301   # Longitudine angolo superiore destro


# Configurazione percorsi di output
geojsonFolder = os.path.join(cartellaBase, nome_esecuzione, "geojson_folder")
outputFolderImages = os.path.join(cartellaBase, nome_esecuzione, "images")
outputFolderAnnotationsImage = os.path.join(cartellaBase, nome_esecuzione, "annotations_image")
outputFolderBounded = os.path.join(cartellaBase, nome_esecuzione, "bounded_images")
outputFolderCSV = os.path.join(cartellaBase, nome_esecuzione, "signal_catalog")
outputRitagli = os.path.join(cartellaBase, nome_esecuzione, "resized_images")

percorso_esecuzione = os.path.join(cartellaBase, nome_esecuzione)
geojson_file_path = os.path.join(geojsonFolder, "mapF_id_list.txt")

percorso_configurazione = BASE_DIR / 'custom_config.txt'
with open(percorso_configurazione, 'r') as f:
    custom_signals = [line.strip() for line in f]

data = Dataminer()
data.chooseConfiguration(Type.CUSTOM)

#Impostazione righe e colonne in cui suddividere l'area
rows = 6
cols = 12

def stampa_progresso(regione, start_time):
    tempo_trascorso = time.time() - start_time
    print(f"\nInizio elaborazione: {regione['nome']}")
    print(f"Bounding Box:")
    print(f"    LAT: {regione['ll_lat']:.6f} → {regione['ur_lat']:.6f}")
    print(f"    LON: {regione['ll_lon']:.6f} → {regione['ur_lon']:.6f}")
    print(f"    Tempo trascorso: {tempo_trascorso:.1f}s")

if os.path.exists(percorso_esecuzione):
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
                while True:   # Chiedi all'utente il numero di feature per file
                    try:
                        num_features_per_file = int(input(f"Inserisci il numero di feature da estrarre da OGNI file GeoJSON: "))
                        if num_features_per_file >= 0:
                            break
                        else:
                            print("Valore non valido. Inserisci un numero non negativo.")
                    except ValueError:
                        print("Input non valido. Inserisci un numero.")

                for geojsonFilePath in geojsonFilePathList:  # Itera sui file GeoJSON
                    n = data.getNGeojson(geojsonFilePath)
                    if n > 0:
                        data.setSelector(number=min(num_features_per_file, n))
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
            while True:  # Chiedi all'utente il numero di feature per file
                try:
                    num_features_per_file = int(
                        input(f"Inserisci il numero di feature da estrarre da OGNI file GeoJSON: "))
                    if num_features_per_file >= 0:
                        break
                    else:
                        print("Valore non valido. Inserisci un numero non negativo.")
                except ValueError:
                    print("Input non valido. Inserisci un numero.")

            for geojsonFilePath in geojsonFilePathList:  # Itera sui file GeoJSON
                n = data.getNGeojson(geojsonFilePath)
                if n > 0:
                    data.setSelector(number=min(num_features_per_file, n))  # Imposta il numero di feature da estrarre
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
    while True:
        try:
            num_features_per_file = int(
                input(f"Inserisci il numero di feature da estrarre da OGNI file GeoJSON: "))
            if num_features_per_file >= 0:
                break
            else:
                print("Valore non valido. Inserisci un numero non negativo.")
        except ValueError:
            print("Input non valido. Inserisci un numero.")
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
            data.setSelector(number=min(num_features_per_file, n))  # Imposta il numero di feature da estrarre
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