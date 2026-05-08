import os
import cv2
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# CONFIGURATION
# =========================================================

training_path = r"C:\Users\ADMIN\Documents\BreathIA\balanced_dataset"
validation_path = r"C:\Users\ADMIN\Documents\BreathIA\Validation_Segments_wav"

IMG_SIZE = 224

sr = 22050
n_mels = 128
n_fft = 2048
hop_length = 512

# =========================================================
# FUNCTION:
# CREATE MEL-SPECTROGRAM IMAGE
# =========================================================

def mel_to_image(y, sr):

    # =============================================
    # MEL SPECTROGRAM
    # =============================================

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # =============================================
    # NORMALIZE 0-255
    # =============================================

    img = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    img = (img * 255).astype(np.uint8)

    # =============================================
    # RESIZE
    # =============================================

    resized_img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    # =============================================
    # CONVERT TO RGB
    # =============================================

    import matplotlib.cm as cm

    colored = cm.jet(resized_img / 255.0)

    rgb_img = colored[:, :, :3]

    # Normalize to 0-1
    rgb_img = rgb_img.astype(np.float32) / 255.0

    return rgb_img

# =========================================================
# FUNCTION:
# LOAD DATASET
# =========================================================

def load_dataset(folder_path):

    X = []
    y_labels = []

    print(f"\nProcessing folder: {folder_path}")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(".wav"):
            continue

        filepath = os.path.join(folder_path, filename)

        try:

            # =========================================
            # LOAD AUDIO
            # =========================================

            y_audio, _ = librosa.load(filepath, sr=sr)

            # =========================================
            # CREATE MEL IMAGE
            # =========================================

            mel_image = mel_to_image(y_audio, sr)

            # =========================================
            # EXTRACT LABEL
            # =========================================

            # concat_000002_Crackle.wav

            label = filename.replace(".wav", "").split("_")[-1]

            # =========================================
            # SAVE
            # =========================================

            X.append(mel_image)
            y_labels.append(label)

            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return np.array(X), np.array(y_labels)

# =========================================================
# LOAD TRAINING DATA
# =========================================================

X_train, y_train_labels = load_dataset(training_path)

# =========================================================
# LOAD VALIDATION DATA
# =========================================================

X_val, y_val_labels = load_dataset(validation_path)

# =========================================================
# LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

all_labels = np.concatenate([y_train_labels, y_val_labels])

label_encoder.fit(all_labels)

y_train = label_encoder.transform(y_train_labels)
y_val = label_encoder.transform(y_val_labels)

# One-hot encoding
num_classes = len(label_encoder.classes_)

y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)

print("\nClasses:")
print(label_encoder.classes_)

# =========================================================
# LOAD EFFICIENTNETB0
# =========================================================

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze layers
for layer in base_model.layers[:-20]:
    layer.trainable = False

for layer in base_model.layers[-20:]:
    layer.trainable = True

# =========================================================
# CUSTOM CLASSIFIER
# =========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(256, activation='relu')(x)

x = Dropout(0.3)(x)

predictions = Dense(
    num_classes,
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# TRAIN
# =========================================================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16,
    callbacks=[early_stop]
)

# =========================================================
# SAVE MODEL
# =========================================================

model.save("efficientnet_b0_breathia.keras")

print("\nModel saved as efficientnet_b0_breathia.keras")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# PREDICTIONS
# =========================================================

y_pred_probs = model.predict(X_val)

# Convert probabilities to class index
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert one-hot validation labels
y_true = np.argmax(y_val, axis=1)

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

class_names = label_encoder.classes_

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

plt.xlabel("Predicted")
plt.ylabel("True")

# Numbers inside cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha='center',
            va='center'
        )

plt.tight_layout()

plt.show()

