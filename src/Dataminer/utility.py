import base64
import mapbox_vector_tile
import cv2
import json
import os
import pandas as pd
import folium
import shutil
import csv


feature_mapping = {
"regulatory--maximum-speed-limit-20" : "Speed limit (20km/h)",
"regulatory--maximum-speed-limit-30" : "Speed limit (30km/h)",
"regulatory--maximum-speed-limit-50" : "Speed limit (50km/h)",
"regulatory--maximum-speed-limit-60" : "Speed limit (60km/h)",
"regulatory--maximum-speed-limit-70" : "Speed limit (70km/h)",
"regulatory--maximum-speed-limit-80" : "Speed limit (80km/h)",
"regulatory--maximum-speed-limit-100" : "Speed limit (100km/h)",
"regulatory--maximum-speed-limit-120" : "Speed limit (120km/h)",
"regulatory--end-of-maximum-speed-limit-80" : "End of speed limit (80km/h)",
"regulatory--no-overtaking" : "No Passing",
"regulatory--no-overtaking-by-heavy-goods-vehicles" : "No passing for vehicles over 3.5 metric tons",
"warning--crossroads" : "Right-of-way at the next intersection",
"regulatory--priority-road" : "Priority road",
"regulatory--yield" : "Yield",
"regulatory--stop" : "Stop",
"regulatory--road-closed-to-vehicles" : "No vehicles",
"regulatory--no-heavy-goods-vehicles" : "Vehicles over 3.5 metric tons prohibited",
"regulatory--no-entry" : "No entry",
"warning--other-danger" : "General caution",
"warning--curve-left" : "Dangerous curve to the left",
"warning--curve-right" : "Dangerous curve to the right",
"warning--double-curve-first-left" : "Double curve",
"warning--road-bump" : "Bumpy road",
"warning--slippery-road-surface" : "Slippery road",
"warning--road-narrows-right" : "Road narrows on the right",
"warning--roadworks" : "Road work",
"warning--traffic-signals" : "Traffic signals",
"warning--pedestrians-crossing" : "Pedestrians",
"warning--children" : "Children crossing",
"warning--bicycles-crossing" : "Bicycles crossing",
"warning--icy-road" : "Beware of ice/snow",
"warning--wild-animals" : "Wild animals crossing",
"regulatory--end-of-maximum-speed-limit" : "End of all speed and passing limits",
"regulatory--turn-right-ahead" : "Turn right ahead",
"regulatory--turn-left-ahead" : "Turn left ahead",
"regulatory--go-straight" : "Ahead only",
"regulatory--go-straight-or-turn-right" : "Go straight or right",
"regulatory--go-straight-or-turn-left" : "Go straight or left",
"regulatory--keep-right" : "Keep right",
"regulatory--keep-left" : "Keep left",
"regulatory--roundabout" : "Roundabout mandatory",
"regulatory--end-of-no-overtaking" : "End of no passing",
"regulatory--end-of-no-overtaking-by-heavy-goods-vehicles" : "End of no passing by vehicles over 3.5 metric tons",
}
def folder_maker(cartellaBase, nome_esecuzione):
    cartella_esecuzione = os.path.join(cartellaBase, nome_esecuzione)
    if not os.path.exists(cartella_esecuzione):
        os.makedirs(cartella_esecuzione)
        print(f"Cartella creata: {cartella_esecuzione}")
    for cartella in ["geojson_folder", "images", "annotations_image",
                     "bounded_images", "signal_catalog", "resized_images"]:
        cartella_completa = os.path.join(cartella_esecuzione, cartella)
        if not os.path.exists(cartella_completa):
            os.makedirs(cartella_completa)
            print(f"Cartella creata: {cartella_completa}")

def get_bbox_coordinates(detection, image_shape):
  """Decodifica la geometria e calcola le coordinate del bounding box."""
  base64_string = detection['geometry']
  decoded_data = base64.decodebytes(base64_string.encode('utf-8'))
  detection_geometry = mapbox_vector_tile.decode(decoded_data)
  coordinates = detection_geometry['mpy-or']['features'][0]['geometry']['coordinates'][0]
  min_x = int(min(coord[0] for coord in coordinates) / 4096 * image_shape[1])
  max_x = int(max(coord[0] for coord in coordinates) / 4096 * image_shape[1])
  min_y = image_shape[0] - int(min(coord[1] for coord in coordinates) / 4096 * image_shape[0])
  max_y = image_shape[0] - int(max(coord[1] for coord in coordinates) / 4096 * image_shape[0])

  if min_y > max_y:
      min_y, max_y = max_y, min_y

  return min_x, max_x, min_y, max_y


def bounding(images, json_folder, output_folder):
    for filename in os.listdir(json_folder):
        if filename.endswith('.json'):
            nome_file = os.path.splitext(filename)[0]
            percorso_json = os.path.join(json_folder, filename)
            percorso_immagine = os.path.join(images, nome_file + '.jpg')

            with open(percorso_json, 'r') as f:
                annotation_data = json.load(f)

            image_data = annotation_data["image"]  # Accedi alle informazioni sull'immagine
            detections = image_data['detections']
            image = cv2.imread(percorso_immagine)


            for detection in detections:
                min_x, max_x, min_y, max_y = get_bbox_coordinates(detection, image.shape)
                cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            output_path = os.path.join(output_folder, nome_file + '_bbox.jpg')
            cv2.imwrite(output_path, image)


def csv_maker(cartella_json, cartella_output, nome_esecuzione):
    for filename in os.listdir(cartella_json):
        if filename.endswith('.json'):
            percorso_json = os.path.join(cartella_json, filename)
            with open(percorso_json, 'r') as f:
                data = json.load(f)

            info_map_feature = data["map_feature"]  # Accedi alle informazioni sul Map Feature
            info_immagine = data["image"]  # Accedi alle informazioni sull'immagine
            detections = info_immagine['detections']

            data_list = []
            for detection in detections:
                data_dict = {
                    'id_map_feature': info_map_feature['id_map_feature'],
                    'object_value': info_map_feature['value'],
                    'geometry_map_feature': info_map_feature['geometry_map_feature'],
                    'id_image': info_immagine['id_image'],
                    'distance(m)': info_immagine['distance(m)'],
                    'geographic_location': info_immagine['geographic_location'],
                    'geometry_polygon': info_immagine.get('geometry_polygon'),
                    'value': detection['value'],
                    'geometry': detection['geometry'],
                    'decoded_geometry': detection['decoded_geometry'],
                    'origins_dataset': nome_esecuzione,
                }
                data_list.append(data_dict)

            df = pd.DataFrame(data_list)
            nome_file_csv = os.path.splitext(filename)[0] + '.csv'
            percorso_csv = os.path.join(cartella_output, nome_file_csv)
            df.to_csv(percorso_csv, index=False)

def resizer(images, input_json, output_folder, custom_signals, percorso_esecuzione):
    """Ritaglia, salva le immagini e crea un CSV con le informazioni sulle feature."""

    csv_data = []  # Lista per i dati del CSV

    for filename in os.listdir(images):
        if filename.endswith(".jpg") and filename != ".DS_Store":
            image_path = os.path.join(images, filename)

            if not os.path.exists(image_path):  # Controlla se il file esiste
                print(f"File immagine non trovato: {image_path}")
                continue

            image = cv2.imread(image_path)
            if image is None:  # Gestisci il ritorno None da imread
                print(f"Errore durante il caricamento dell'immagine: {image_path}")
                continue

            print(f"Elaborazione immagine: {filename}")
            nome_immagine = os.path.splitext(filename)[0]
            percorso_json = os.path.join(input_json, nome_immagine + '.json')

            if not os.path.exists(percorso_json):
                print(f"File JSON non trovato: {percorso_json}")
                continue

            try:
                with open(percorso_json, 'r') as f:
                    annotation_data = json.load(f)
                    detections = annotation_data["image"]['detections']

                for i, detection in enumerate(detections):
                    if detection['value'] in custom_signals:
                        try:
                            min_x, max_x, min_y, max_y = get_bbox_coordinates(detection, image.shape)
                            cropped_image = image[min_y:max_y, min_x:max_x]
                            output_path = os.path.join(output_folder, f"{nome_immagine}_{i}.jpg")
                            cv2.imwrite(output_path, cropped_image)

                            # Aggiungi i dati al CSV
                            parts = detection['value'].split('--')  # Divide il nome in parti
                            feature_name = '--'.join(parts[:-1])  # Unisci tutti gli elementi tranne l'ultimo

                            # Aggiungi i dati al CSV con il nome personalizzato
                            csv_data.append([f"{nome_immagine}_{i}.jpg",
                                             feature_mapping.get(feature_name, feature_name)])

                        except (cv2.error, ValueError, TypeError) as e:  # Gestisci errori di cv2, get_bbox e TypeError
                            print(f"Errore durante il ritaglio/salvataggio: {e}")
                            print(f"Immagine: {filename}, Detection: {i}")
                            print(f"Coordinate: min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}") # Debug coordinates

            except (KeyError, json.JSONDecodeError) as e:  # Gestisci errori JSON
                print(f"Errore nel file JSON {percorso_json}: {e}")

    # Scrivi i dati nel file CSV
    csv_path = os.path.join(percorso_esecuzione, "annotations.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["filename", "feature"])  # Intestazione
        csv_writer.writerows(csv_data)

    print(f"File CSV creato: {csv_path}")

def map_maker(cartella_json, ll_lat, ll_lon, ur_lat, ur_lon, nome_esecuzione, cartellaBase):
    """
    Crea una mappa con i marker delle immagini e la salva nella cartella dell'esecuzione.

    Args:
        cartella_json (str): Percorso della cartella contenente i file JSON delle annotazioni.
        ll_lat (float): Latitudine dell'angolo inferiore sinistro della bounding box.
        ll_lon (float): Longitudine dell'angolo inferiore sinistro della bounding box.
        ur_lat (float): Latitudine dell'angolo superiore destro della bounding box.
        ur_lon (float): Longitudine dell'angolo superiore destro della bounding box.
        nome_esecuzione (str): Nome dell'esecuzione corrente.
    """
    latitudine_centro = (ll_lat + ur_lat) / 2
    longitudine_centro = (ll_lon + ur_lon) / 2
    my_map = folium.Map(location=[latitudine_centro, longitudine_centro], zoom_start=5)

    for filename in os.listdir(cartella_json):
        if filename.endswith('.json'):
            percorso_json = os.path.join(cartella_json, filename)
            with open(percorso_json, 'r') as f:
                data = json.load(f)
            info_immagine = data["image"]
            latitudine = info_immagine['geometry_image']['coordinates'][1]
            longitudine = info_immagine['geometry_image']['coordinates'][0]
            folium.Marker(location=[latitudine, longitudine], popup=f"Immagine: {filename}").add_to(my_map)

    # Crea il percorso per il file HTML della mappa nella cartella dell'esecuzione
    percorso_mappa = os.path.join(cartellaBase, nome_esecuzione, f"{nome_esecuzione}_immagini.html")

    my_map.save(percorso_mappa)
    print(f"Mappa salvata in: {percorso_mappa}")


def check_files(text_file, image_folder):
    """
    Verifica la presenza di file .jpg in una cartella in base a una lista in un file di testo.

    Args:
        text_file: Percorso del file di testo contenente la lista di oggetti.
        image_folder: Percorso della cartella contenente i file .jpg.
        output_file: Percorso del file di testo in cui scrivere i codici non trovati.
    """

    try:
        with open(text_file, 'r') as f:
            objects = [line.strip() for line in f]  # Leggi gli oggetti dal file, rimuovendo spazi bianchi

    except FileNotFoundError:
        print(f"Errore: File di testo '{text_file}' non trovato.")
        return

    if not os.path.isdir(image_folder):
        print(f"Errore: Cartella immagini '{image_folder}' non trovata.")
        return

    missing_objects = []

    for obj in objects:
        image_file = os.path.join(image_folder, obj + ".jpg")  # Costruisci il nome del file immagine
        if not os.path.isfile(image_file):
            missing_objects.append(obj)

    return missing_objects


def map_maker2(cartella_json, nome_esecuzione, cartellaBase):
    """
    Crea una mappa con i marker delle immagini basandosi sui dati JSON.

    Args:
        cartella_json (str): Percorso della cartella contenente i file JSON.
        nome_esecuzione (str): Nome dell'esecuzione.
        cartellaBase (str): Percorso della cartella base.
    """

    # Lista per memorizzare tutte le coordinate
    coordinates = []

    for filename in os.listdir(cartella_json):
        if filename.endswith('.json'):
            percorso_json = os.path.join(cartella_json, filename)
            try:
                with open(percorso_json, 'r') as f:
                    data = json.load(f)
                    lat = data['image']['geometry_image']['coordinates'][1]
                    lon = data['image']['geometry_image']['coordinates'][0]
                    coordinates.append([lat, lon])
            except (KeyError, IndexError): # gestisci file json formattati male
                print(f"Errore nella lettura delle coordinate da {filename}. File saltato.")
                continue # salta il file se ci sono errori

    if not coordinates: # gestisci il caso in cui non ci sono coordinate valide
        print("Nessuna coordinata trovata nei file JSON. Impossibile creare la mappa.")
        return

    # Calcola il centro della mappa come media delle latitudini e longitudini
    avg_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
    avg_lon = sum(coord[1] for coord in coordinates) / len(coordinates)


    # Crea la mappa centrata sulle coordinate medie
    my_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=5) # zoom_start dinamico dopo

    # Aggiungi i marker alla mappa
    for coord in coordinates:
        folium.Marker(location=coord, popup=f"Lat: {coord[0]}, Lon: {coord[1]}").add_to(my_map)


    percorso_mappa = os.path.join(cartellaBase, nome_esecuzione, f"{nome_esecuzione}_immagini.html")
    my_map.save(percorso_mappa)
    print(f"Mappa salvata in: {percorso_mappa}")

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
