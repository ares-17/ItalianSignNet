import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras import Sequential
import json
import pandas as pd
# Parametri del modello e training
MODEL_PATH = 'Saved_Models/VGGnet.keras'  # Percorso per salvare/caricare il modello
EPOCHS = 15
BATCH_SIZE = 64

# Mapping ClassID to traffic sign names
signs = []
with open('../predictionToolV2/signnames.csv', 'r') as csvfile:
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


# Carica il modello se esiste, altrimenti addestralo
if os.path.exists(MODEL_PATH):
    print("Caricamento del modello da:", MODEL_PATH)
    VGGNet_Model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Addestramento del modello...")

    training_file = "../predictionToolV2/traffic-signs-data/train.p"
    validation_file = "../predictionToolV2/traffic-signs-data/valid.p"
    testing_file = "../predictionToolV2/traffic-signs-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    # Number of training examples
    n_train = X_train.shape[0]

    # Number of testing examples
    n_test = X_test.shape[0]

    # Number of validation examples.
    n_validation = X_valid.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))

    print("Number of training examples: ", n_train)
    print("Number of testing examples: ", n_test)
    print("Number of validation examples: ", n_validation)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    # Preprocessing
    X_train_preprocessed = preprocess(X_train)
    X_valid_preprocessed = preprocess(X_valid)
    X_test_preprocessed = preprocess(X_test)

    # Definizione del modello VGGnet usando Sequential
    VGGNet_Model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='SAME', input_shape=(32, 32, 1)),
        Conv2D(32, (3, 3), activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu', padding='SAME'),
        Conv2D(64, (3, 3), activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        Conv2D(128, (3, 3), activation='relu', padding='SAME'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(n_classes)  # Utilizza n_classes calcolato dai dati
    ])

    VGGNet_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                         loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])

    # Addestramento del modello
    history = VGGNet_Model.fit(X_train_preprocessed, tf.keras.utils.to_categorical(y_train),
                               epochs=EPOCHS, batch_size=BATCH_SIZE,
                               validation_data=(X_valid_preprocessed, tf.keras.utils.to_categorical(y_valid)))

    # Salvataggio del modello
    VGGNet_Model.save(MODEL_PATH)
    print("Modello salvato in:", MODEL_PATH)

    # Valutazione del modello sul test set
    test_loss, test_accuracy = VGGNet_Model.evaluate(X_test_preprocessed,
                                                     tf.keras.utils.to_categorical(y_test), verbose=0)
    print("Test Accuracy = {:.1f}%".format(test_accuracy * 100))

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.show()

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

    # Ordina le predizioni per nome file
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
        merged_df.to_csv(output_path, index=False)
        print(f"File CSV uniti con successo e salvati in {output_path}")

    except FileNotFoundError as e:
        print(f"Errore: File non trovato: {e}")
    except pd.errors.ParserError as e:
        print(f"Errore durante la lettura del CSV: {e}")
    except ValueError as e:
        print(f"Errore: {e}")
    except Exception as e:
        print(f"Errore generico durante il merge: {e}")  # Cattura altri potenziali errori



image_folder = '/Users/matteospavone/Desktop/Testing/Test2/resized_images'
json_folder = '/Users/matteospavone/Desktop/Testing/Test2/annotations_image' # Percorso della cartella JSON
predict_and_save_to_csv(VGGNet_Model, image_folder, json_folder)

#Sostituisci con il percorso del tuo file che contiene le predizioni
csv1_path = "/Users/matteospavone/Desktop/Pycharm/predictionToolV2/predictions.csv"  

#Sostituisci con il percorso del file annotations.csv generato del tuo dataset
csv2_path = "/Users/matteospavone/Desktop/Testing/Test2/annotations.csv"  


output_path = "merged_file.csv" # Sostituisci con il percorso desiderato per il file di output
merge_csv_on_filename(csv1_path, csv2_path, output_path)

