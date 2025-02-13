import os
import shutil
import sys
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path

# Aggiungi il percorso della cartella "utils" al PYTHONPATH
utils_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(utils_path)

import dataminer.utility as utility

load_dotenv()
BASE_DIR = Path(os.getenv("BASE_DIR"))

# Boolean options
MERGE_FULLSIZE_IMAGES = os.getenv("MERGE_FULLSIZE_IMAGES", "false").lower() in ("true", "1", "yes")
MERGE_BOUNDED_IMAGES = os.getenv("MERGE_BOUNDED_IMAGES", "false").lower() in ("true", "1", "yes")

def merge_test_data_folders(input_folders, input_directory, output_folder):
    """
    Unisce le cartelle Test_data specificate, inclusi i file annotations.csv.
    """
    subfolders = ["annotations_image", "geojson_folder", "resized_images", "signal_catalog"]

    if MERGE_FULLSIZE_IMAGES:
        subfolders.append("images")

    if MERGE_BOUNDED_IMAGES:
        subfolders.append("bounded_images")

    output_folder_path = os.path.join(input_directory, output_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    for subfolder in subfolders:
        os.makedirs(os.path.join(output_folder_path, subfolder), exist_ok=True)

    all_annotations = []
    for folder_name in input_folders:
        source_folder = os.path.join(input_directory, folder_name)

        if not os.path.exists(source_folder):
            print(f"Errore: la cartella {folder_name} non esiste. Saltando.")
            continue

        for subfolder in subfolders:
            source_path = os.path.join(source_folder, subfolder)
            destination_path = os.path.join(output_folder_path, subfolder)

            if os.path.exists(source_path):
                for item in os.listdir(source_path):
                    s = os.path.join(source_path, item)
                    d = os.path.join(destination_path, item)

                    try:
                        if os.path.isdir(s):
                            shutil.copytree(s, d, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s, d)
                    except Exception as e:
                        print(f"Errore nel copiare {s} → {d}: {e}")

        annotations_path = os.path.join(source_folder, "annotations.csv")
        if os.path.exists(annotations_path):
            try:
                df = pd.read_csv(annotations_path)
                if not df.empty:
                    all_annotations.append(df)
                else:
                    print(f"Avviso: Il file annotations.csv in {folder_name} è vuoto.")
            except pd.errors.EmptyDataError:
                print(f"Avviso: Errore nel leggere annotations.csv in {folder_name} (file vuoto o corrotto).")
        else:
            print(f"Avviso: Il file annotations.csv non è presente in {folder_name}.")

    if all_annotations:
        try:
            combined_annotations = pd.concat(all_annotations, ignore_index=True)
            combined_annotations.to_csv(os.path.join(output_folder_path, "annotations.csv"), index=False)
            print("File annotations.csv uniti con successo.")
        except Exception as e:
            print(f"Errore nella fusione del file annotations.csv: {e}")



def list_available_folders(directory="."):
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    return folders


input_directory = BASE_DIR / "testing"

try:
    available_folders = list_available_folders(input_directory)
    print("Dataset disponibili:")
    if available_folders:
        for i, folder in enumerate(available_folders):
            print(f"{i + 1}. {folder}")

        # Chiedi all'utente di selezionare le cartelle
        selected_indices = []
        while True:
            try:
                user_input = input(
                    "\nInserisci i numeri delle cartelle da unire (separati da virgola, o 'all' per effettuare il merge di tutte le cartelle presenti): ")
                if user_input.lower() == "all":
                    selected_indices = list(range(len(available_folders)))
                    break

                selected_indices = [int(x.strip()) for x in user_input.split(",")]
                if all(1 <= i <= len(available_folders) for i in selected_indices):
                    break  # Input valido
                else:
                    print("Numeri di cartella non validi. Riprova.")
            except ValueError:
                print("Input non valido. Inserisci numeri separati da virgola.")

        folders_to_merge = [available_folders[i - 1] for i in selected_indices]


        # Chiedi il nome della cartella di output e controlla la sua esistenza
        while True:
            output_folder = input("Inserisci il nome della cartella di output: ")
            output_folder_path = os.path.join(input_directory, output_folder)
            if os.path.exists(output_folder_path):
                print(f"Errore: la cartella '{output_folder}' esiste già. Scegli un nome diverso.")
            else:
                break

    else:
        print("Nessuna cartella trovata nella directory di input specificata.")
        exit()

    merge_test_data_folders(folders_to_merge, input_directory, output_folder)
    print(f"Cartelle unite con successo in {os.path.join(input_directory, output_folder)}")

    outputFolderAnnotationsImage = os.path.join(input_directory, output_folder, "annotations_image")
    utility.map_maker2(outputFolderAnnotationsImage, output_folder, input_directory)

except (ValueError, FileNotFoundError) as e:  # Gestisci potenziali errori
    print(f"Errore: {e}")
