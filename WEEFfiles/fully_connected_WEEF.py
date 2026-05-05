import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import librosa
import os

# ================= PATHS =================
train_folders = [
    r"C:\Users\ADMIN\Documents\BreathIA\Augmented_Segments_wav_4",
    r"C:\Users\ADMIN\Documents\BreathIA\Segments_wav_4"
]

val_folder = r"C:\Users\ADMIN\Documents\BreathIA\Validation_Segments_wav"

# ================= PARAMETERS =================
segment_length_sec = 3
sr = 22050
N_MFCC = 64

# ================= FUNCTION =================
def process_folders(folders):
    processed = []

    for folder in folders:
        print(f"Procesando carpeta: {folder}")

        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                filepath = os.path.join(folder, filename)

                try:
                    label = filename.replace(".wav", "").split('_')[-1]
                except:
                    label = "unknown"

                try:
                    y, _ = librosa.load(filepath, sr=sr)

                    samples_per_segment = int(segment_length_sec * sr)
                    num_segments = len(y) // samples_per_segment

                    for i in range(num_segments):
                        start = i * samples_per_segment
                        end = start + samples_per_segment
                        segment = y[start:end]

                        if len(segment) == 0:
                            continue

                        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)

                        if mfccs.shape[1] > 0:
                            mfccs_processed = np.mean(mfccs.T, axis=0)

                            if len(mfccs_processed) == N_MFCC:
                                processed.append((mfccs_processed, label))

                except Exception as e:
                    print(f"Error en {filename}: {e}")

    return processed


# ================= LOAD DATA =================
print("=== TRAIN DATA ===")
train_data = process_folders(train_folders)

print("=== VALIDATION DATA ===")
val_data = process_folders([val_folder])

# ================= SPLIT FEATURES =================
X_train = np.array([x[0] for x in train_data])
y_train_labels = [x[1] for x in train_data]

X_val = np.array([x[0] for x in val_data])
y_val_labels = [x[1] for x in val_data]

print(f"Train samples: {X_train.shape}")
print(f"Validation samples: {X_val.shape}")

# ================= ENCODING =================
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_labels)
y_val = encoder.transform(y_val_labels)

print("Clases:", encoder.classes_)

# ================= SCALING =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ================= MODEL =================
model = keras.Sequential([
    keras.layers.Input(shape=(N_MFCC,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================= TRAIN =================
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=200,
    batch_size=128,
    validation_data=(X_val_scaled, y_val)  # <-- REAL validation
)

print("\nEntrenamiento finalizado")

# ================= EVALUATION =================
loss, accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

# ================= CONFUSION MATRIX =================
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = np.argmax(model.predict(X_val_scaled), axis=1)

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()