import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout
from tensorflow.keras import Sequential
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("BASE_DIR")
EPOCHS = 15
BATCH_SIZE = 64

current_directory = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_directory, 'saved', 'VGGnet.keras')

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

training_file = "../model/traffic-signs-data/train.p"
validation_file = "../model/traffic-signs-data/valid.p"
testing_file = "../model/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = X_train.shape[0]
n_test = X_test.shape[0]
n_validation = X_valid.shape[0]
image_shape = X_train[0].shape
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
    Dense(n_classes)
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