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
def process_train(folders):
    processed = []

    for folder in folders:
        print(f"Procesando TRAIN: {folder}")

        for filename in os.listdir(folder):
            if filename.endswith(".wav"):
                filepath = os.path.join(folder, filename)

                label = filename.replace(".wav", "").split('_')[-1]

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

                        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                        mfcc = np.mean(mfcc.T, axis=0)

                        if len(mfcc) == N_MFCC:
                            processed.append((mfcc, label))

                except Exception as e:
                    print(f"Error en {filename}: {e}")

    return processed

def process_validation(folder):
    data = []

    print(f"Procesando VALIDATION: {folder}")

    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder, filename)

            label = filename.replace(".wav", "").split('_')[-1]

            try:
                y, _ = librosa.load(filepath, sr=sr)

                samples_per_segment = int(segment_length_sec * sr)

                # 🔥 sliding windows
                step = samples_per_segment  # no overlap (you can change later)

                file_segments = []

                for start in range(0, len(y), step):
                    end = start + samples_per_segment
                    segment = y[start:end]

                    # pad last segment if needed
                    if len(segment) < samples_per_segment:
                        segment = np.pad(segment, (0, samples_per_segment - len(segment)))

                    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC)
                    mfcc = np.mean(mfcc.T, axis=0)

                    if len(mfcc) == N_MFCC:
                        file_segments.append(mfcc)

                if len(file_segments) > 0:
                    data.append((file_segments, label))

            except Exception as e:
                print(f"Error en {filename}: {e}")

    return data

# ================= LOAD DATA =================
print("=== TRAIN DATA ===")
train_data = process_train(train_folders)

print("=== VALIDATION DATA ===")
val_data = process_validation(val_folder)

# ================= SPLIT FEATURES =================
X_train = np.array([x[0] for x in train_data])
y_train_labels = [x[1] for x in train_data]

X_val_files = [x[0] for x in val_data]  # list of segments per file
y_val_labels = [x[1] for x in val_data]

print(f"Train samples: {X_train.shape}")
print(f"Validation files: {len(X_val_files)}")

# ================= ENCODING =================
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train_labels)
y_val = encoder.transform(y_val_labels)

print("Clases:", encoder.classes_)



# ================= SCALING =================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


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
    validation_split=0.1
)
print("\nEntrenamiento finalizado")

# ================= VALIDATION (FILE-LEVEL) =================
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

y_pred_files = []

for file_segments in X_val_files:
    # scale each segment
    segments_scaled = scaler.transform(file_segments)

    # predict probabilities
    preds = model.predict(segments_scaled, verbose=0)

    # convert to class indices
    pred_classes = np.argmax(preds, axis=1)

    # majority vote
    final_pred = Counter(pred_classes).most_common(1)[0][0]
    y_pred_files.append(final_pred)

# ================= METRICS =================
acc = accuracy_score(y_val, y_pred_files)
print(f"Validation Accuracy (file-level): {acc:.4f}")

# ================= CONFUSION MATRIX =================
cm = confusion_matrix(y_val, y_pred_files)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (File-level)')
plt.show()
