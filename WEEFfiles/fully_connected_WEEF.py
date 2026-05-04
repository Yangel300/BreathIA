import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import librosa
import os

audio_folder = 'folder/entrada' # <--- Ruta a los .wav

# DURACIÓN DEL SEGMENTO 
segment_length_sec = 3
sr = 22050 #Ponerlo desde el inicio

processed_audio_data = {}

print(f"Iniciando procesamiento")

if not os.path.exists(audio_folder):
    print(f"Error: La carpeta '{audio_folder}' no existe. Cambiar el path.")
else:
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(audio_folder, filename)

            # Extracción del nombre (toca ver que)
            try:
                label = filename.split('_')[0] # Toca cambiarlo dependiendo de la estructura del nombre
            except IndexError:
                label = "unknown_label" # Error

            try:
                # Cargar el archivo de audio con sr
                y, current_sr = librosa.load(filepath, sr=sr)

                # Calcular segementos de 3 segundos (no sé cómo estaba pensado para la transformación)
                samples_per_segment = int(segment_length_sec * current_sr)

                # Segmentar el audio
                num_segments = len(y) // samples_per_segment

                segments_list = []
                for i in range(num_segments):
                    start_sample = i * samples_per_segment
                    end_sample = start_sample + samples_per_segment
                    segment = y[start_sample:end_sample]

                    segments_list.append({
                        'segment': segment,
                        'event_type': label
                    })

                processed_audio_data[filename] = segments_list
                print(f"Se han procesado {len(segments_list)} segmentos de {filename} (label: '{label}').")

            except Exception as e:
                print(f"Error procesando {filename}: {e}")

print(f"\nCompletado. Total de audios: {len(processed_audio_data)}")
total_segments_count = sum(len(v) for v in processed_audio_data.values())
print(f"Segmentos de 'processed_audio_data': {total_segments_count}")

# ENTRADA --> processed_audio_data

# FEATURES Y LABELS

# Parámetros para la red
N_MFCC = 64 # Neuronas de entrada
SR = 22050 # Sample rate
MAX_PAD_LEN = 175 # No estoy muy seguro de qué cambia este parámetro

features = []
labels = []

print("Extrayendo features de la entrada...")
for audio_file_name, segments_list in processed_audio_data.items():
    for segment_info in segments_list:
        audio_segment = segment_info['segment']
        event_type = segment_info['event_type']

        # Asegurar que no sea vacío el segmento
        if len(audio_segment) == 0:
            continue

        # Extract MFCCs (Mel-frequency cepstral coefficients) (para esto se usa N_MFCC)
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=SR, n_mfcc=N_MFCC)

        # Esta parte no se si sea necesaria, me ayudo la IA porque generaba error en el modelo
        if mfccs.shape[1] > 0:
            mfccs_processed = np.mean(mfccs.T, axis=0)
            if len(mfccs_processed) == N_MFCC:
                features.append(mfccs_processed)
                labels.append(event_type)
            else:
                print(f"Warning: MFCCs vector length mismatch ({len(mfccs_processed)} != {N_MFCC}). Skipping.")
        else:
            print(f"Warning: No MFCC frames extracted for a segment. Skipping.")

# Convertir a numpy
X = np.array(features)

print(f"Se extrajeron {X.shape[0]} samples con {X.shape[1]} features.")

# Obtener los labels

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(labels)
num_classes = len(np.unique(y_encoded))
print(encoder.classes_) # Verificar que leyó los nombres bien

#  DATA SPLITTING

# 80% para train y 20% para test

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# ESTANDARIZAR DATOS

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Datos escalados.")

# CREAR MODELO

# Red neuronal fully connected con dos capas ocultas

# Entrada (Dense layer): 64 neuronas, Hidden layer 1: 32 neuronas, Hidden Layer 2: 16 neuronas (siguiendo https://www.nature.com/articles/s41598-025-09524-8.pdf)

model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(N_MFCC,)),  # Input layer de 64 neuronas
    keras.layers.Dense(32, activation='relu'),      # Hidden layer 1 de 32 neuronas y ReLU como función de activación
    keras.layers.Dense(16, activation='relu'),      # Hidden layer 2 de 16 neuronas y ReLU como función de activación
    keras.layers.Dense(num_classes, activation='softmax') # Output layer (usan sigmoide pero en nuestro caso no sirve porque no es binario)
])

# Compilación del modelo
model.compile(
    optimizer='adam', # Usaba el optimizador Adam y Categorical Crossentropy
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'] # Definieron Accuracy, Sensitivity y Specificity
)

model.summary()

# ENTRENAMIENTO

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200 , # Toca hacer pruebas, así está en el paper
    batch_size=128, #
    validation_split=0.1 # Validation 10% de los datos
)

print("\nEntrenamiento finalizado")

# EVALUACIÓN DEL MODELO

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# DIBUJAR LA MATRIZ DE CONFUSIÓN

y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)

# Computar la matriz
cm = confusion_matrix(y_test, y_pred)

# Labels
class_labels = encoder.classes_

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
