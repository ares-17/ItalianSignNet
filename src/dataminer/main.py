import os
import subprocess
from bounding_boxes import BOUNDING_BOXES
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

# Carica le variabili d'ambiente
load_dotenv()

NUM_FEATURES = int(os.getenv("NUM_FEATURES_BBOX"))
DOWNLOAD_BBOX_IMAGES_PATH = os.getenv("DOWNLOAD_BBOX_IMAGES_PATH")
MAX_PARALLEL_EXECUTIONS = int(os.getenv("MAX_PARALLEL_EXECUTIONS", "4"))

def process_region(region_key, bbox):
    region_name = bbox.get("nome", region_key)
    print(f"Avvio elaborazione per la regione: {region_name}")

    args = [
        "python", DOWNLOAD_BBOX_IMAGES_PATH,
        "--ll_lat", str(bbox["ll_lat"]),
        "--ll_lon", str(bbox["ll_lon"]),
        "--ur_lat", str(bbox["ur_lat"]),
        "--ur_lon", str(bbox["ur_lon"]),
        "--num_features", str(NUM_FEATURES),
        "--nome_esecuzione", region_key
    ]
    result = subprocess.run(args)
    if result.returncode != 0:
        print(f"Errore nell'elaborazione della regione {region_name}.")
    else:
        print(f"Elaborazione per {region_name} completata.\n")
    return region_name, result.returncode

def main():
    futures = []
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_EXECUTIONS) as executor:
        for region_key, bbox in BOUNDING_BOXES.items():
            future = executor.submit(process_region, region_key, bbox)
            futures.append(future)
        # Attende il completamento di tutti i processi
        for future in as_completed(futures):
            region_name, rc = future.result()
            if rc != 0:
                print(f"Regione {region_name}: completata con errori.")
            else:
                print(f"Regione {region_name}: elaborazione completata con successo.")

if __name__ == '__main__':
    main()
