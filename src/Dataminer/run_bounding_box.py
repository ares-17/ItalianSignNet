import os
import subprocess
from bounding_boxes import BOUNDING_BOXES

NUM_FEATURES = 10

def main():
    for region_key, bbox in BOUNDING_BOXES.items():
        region_name = bbox.get("nome", region_key)
        print(f"Avvio elaborazione per la regione: {region_name}")

        args = [
            "python", "dowload_with_bounding_box.py",
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

if __name__ == '__main__':
    main()
