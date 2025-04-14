import numpy as np
import cv2
import csv
import os
import tensorflow as tf
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
ALL_SIGNS_NAME_PATH = os.getenv("ALL_SIGNS_NAME_PATH", "all_signs")

testing_directory = os.path.join(BASE_DIR, 'testing')
current_directory = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(current_directory, '.results'), exist_ok=True)
MODEL_PATH = os.path.join(current_directory, 'saved', 'VGGnet.keras')

# Mapping ClassID to traffic sign names
signs = []
with open(os.path.join(BASE_DIR, 'src','utils','signnames.csv'), 'r') as csvfile:
    signnames = csv.reader(csvfile, delimiter=',')
    next(signnames, None)
    for row in signnames:
        signs.append(row[1])
    csvfile.close()

# Funzioni di pre-elaborazione
def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def local_histo_equalize(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def image_normalize(image):
    return image / 255.0

def preprocess(data):
    gray_images = np.array([gray_scale(img) for img in data])
    equalized_images = np.array([local_histo_equalize(img) for img in gray_images])
    normalized_images = np.array([image_normalize(img) for img in equalized_images])
    return normalized_images[..., np.newaxis]


VGGNet_Model = tf.keras.models.load_model(MODEL_PATH)
    
# Funzione per la predizione su una cartella di immagini
def predict_and_save_to_csv(model, image_folder, json_folder, csv_filename="predictions.csv"):
    predictions = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            base_name = filename.split('_')[0] # Estrae il nome base del file
            json_path = os.path.join(json_folder, f"{base_name}.json")

            if os.path.exists(json_path):  # Controlla se il file JSON esiste
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    geo_location = json_data['image']['geographic_location']
            else:
                geo_location = "N/A" # Se il JSON non esiste, geolocalizzazione non disponibile
                print(f"Warning: JSON file not found for {filename}")


            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (32, 32))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            preprocessed_img = preprocess(np.array([img]))[0]

            probability = model.predict(np.array([preprocessed_img]))[0]
            pred = np.argmax(probability)
            class_name = signs[pred]
            class_probability = tf.nn.softmax(probability)[pred].numpy()

            predictions.append([filename, class_name, class_probability, pred, geo_location])

    predictions.sort(key=lambda x: x[0])


    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Predicted Class', 'Probability', 'Class ID', 'Geographic Location'])
        csv_writer.writerows(predictions)

def merge_csv_on_filename(csv1_path, csv2_path, output_path="merged.csv"):
    """Unisce due CSV su 'filename', gestendo differenze di capitalizzazione."""
    try:
        df1 = pd.read_csv(csv1_path)
        df2 = pd.read_csv(csv2_path)

        # Trova il nome corretto della colonna "filename" in ogni DataFrame (case-insensitive)
        filename_col1 = next((col for col in df1.columns if col.lower() == "filename"), None)
        filename_col2 = next((col for col in df2.columns if col.lower() == "filename"), None)

        if not filename_col1:
            raise ValueError(f"Colonna 'filename' (case-insensitive) non trovata in {csv1_path}")
        if not filename_col2:
            raise ValueError(f"Colonna 'filename' (case-insensitive) non trovata in {csv2_path}")

        # Rinomina le colonne temporaneamente per il merge, se necessario
        if filename_col1 != "filename":
            df1 = df1.rename(columns={filename_col1: "filename"})
        if filename_col2 != "filename":
            df2 = df2.rename(columns={filename_col2: "filename"})

        merged_df = pd.merge(df1, df2, on="filename", how="outer")
        
        if "Predicted Class" not in merged_df.columns or "feature" not in merged_df.columns:
            raise ValueError("Le colonne 'Predicted Class' e/o 'feature' non sono presenti nel CSV unito.")
        merged_df["Results"] = merged_df.apply(lambda row: "T" if str(row["Predicted Class"]) == str(row["feature"]) else "F", axis=1)
        
        merged_df.to_csv(output_path, index=False)
        print(f"File CSV uniti con successo e salvati in {output_path}")

    except FileNotFoundError as e:
        print(f"Errore: File non trovato: {e}")
    except pd.errors.ParserError as e:
        print(f"Errore durante la lettura del CSV: {e}")
    except ValueError as e:
        print(f"Errore: {e}")
    except Exception as e:
        print(f"Errore generico durante il merge: {e}")


image_folder = os.path.join(testing_directory, ALL_SIGNS_NAME_PATH, 'resized_images')
json_folder = os.path.join(testing_directory, ALL_SIGNS_NAME_PATH, 'annotations_image')
predict_and_save_to_csv(VGGNet_Model, image_folder, json_folder, os.path.join(current_directory, '.results', 'predictions.csv'))

csv1_path = os.path.join(current_directory, '.results', 'predictions.csv') 
csv2_path = os.path.join(testing_directory, ALL_SIGNS_NAME_PATH, 'annotations.csv') 
output_path = os.path.join(current_directory, '.results', 'merged_file.csv')
merge_csv_on_filename(csv1_path, csv2_path, output_path)

