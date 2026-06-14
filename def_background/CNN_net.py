import os
import random
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import torch
import joblib
from scipy.signal import resample

from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# =========================================================
# GPU CONFIG
# =========================================================

torch.backends.cudnn.benchmark = True

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("\n=================================================")
print(f"DEVICE: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

else:

    print("CUDA NOT AVAILABLE")

print("=================================================\n")

# =========================================================
# PATHS
# =========================================================

training_path = (
    r"C:\Users\Oficina 01\Documents\Breath\BreathIA\augmentation_dataset_with_rules"
)

validation_path = (
    r"C:\Users\Oficina 01\Documents\Breath\BreathIA\validation_dataset_with_rules"
)

# =========================================================
# AUDIO CONFIG
# =========================================================

sr = 22050

duration = 3

target_length = sr * duration

n_mels = 64

batch_size = 8

epochs = 280

# =========================================================
# LABELS
# =========================================================

labels = [
    "Fine_Crackle",
    "Normal",
    "Wheeze"
]

label_encoder = LabelEncoder()

label_encoder.fit(labels)

num_classes = len(labels)

print("Classes:")
print(label_encoder.classes_)

# =========================================================
# DATASET
# =========================================================

class BreathDataset(Dataset):

    def __init__(self, folder):

        self.folder = folder

        self.files = []

        for f in os.listdir(folder):

            if not f.lower().endswith(".wav"):
                continue

            lower = f.lower()

            if (
                "normal" in lower
                or "wheeze" in lower
                or "fine_crackle" in lower
            ):

                self.files.append(f)

        print(
            f"{folder} -> {len(self.files)} files"
        )

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        filepath = os.path.join(
            self.folder,
            filename
        )

        # =================================================
        # LOAD AUDIO
        # =================================================

        waveform, original_sr = sf.read(filepath)

        waveform = waveform.astype(np.float32)

        # Stereo -> mono
        if len(waveform.shape) > 1:

            waveform = np.mean(
                waveform,
                axis=1
            )

        # =================================================
        # RESAMPLE
        # =================================================

        if original_sr != sr:

            new_length = int(
                len(waveform)
                * sr
                / original_sr
            )

            waveform = resample(
                waveform,
                new_length
            )

        # =================================================
        # FIX LENGTH
        # =================================================

        if len(waveform) < target_length:

            waveform = np.pad(
                waveform,
                (
                    0,
                    target_length - len(waveform)
                )
            )

        else:

            waveform = waveform[
                :target_length
            ]

        # =================================================
        # MEL SPECTROGRAM
        # =================================================

        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=2048,
            hop_length=512,
            n_mels=n_mels
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # =================================================
        # NORMALIZE
        # =================================================

        mel_db = (
            mel_db - mel_db.mean()
        ) / (
            mel_db.std() + 1e-8
        )

        # =================================================
        # TO TENSOR
        # =================================================

        mel_db = torch.tensor(
            mel_db,
            dtype=torch.float32
        ).unsqueeze(0)

        # =================================================
        # LABEL
        # =================================================

        lower = filename.lower()

        if "fine_crackle" in lower:

            label_name = "Fine_Crackle"

        elif "wheeze" in lower:

            label_name = "Wheeze"

        elif "normal" in lower:

            label_name = "Normal"

        else:

            raise ValueError(
                f"Unknown label in file: {filename}"
            )

        label = label_encoder.transform(
            [label_name]
        )[0]

        return mel_db, label

# =========================================================
# DATASETS
# =========================================================

train_dataset = BreathDataset(
    training_path
)

val_dataset = BreathDataset(
    validation_path
)

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# =========================================================
# CNN
# =========================================================

class SimpleAudioCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Dropout(0.3)
        )

        # =================================================
        # AUTO FLATTEN
        # =================================================

        dummy = torch.zeros(
            1,
            1,
            n_mels,
            130
        )

        out = self.features(dummy)

        flatten_size = out.reshape(
            1,
            -1
        ).shape[1]

        print(
            f"Flatten size: {flatten_size}"
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                flatten_size,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                256,
                num_classes
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# =========================================================
# MODEL
# =========================================================

model = SimpleAudioCNN().to(device)

print(model)

# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================================================
# TRAIN
# =========================================================

print("\nSTART TRAINING\n")

train_losses = []

train_accuracies = []

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for mel, labels_batch in train_loader:

        mel = mel.to(device)

        labels_batch = labels_batch.to(device)

        # =================================================
        # FORWARD
        # =================================================

        outputs = model(mel)

        loss = criterion(
            outputs,
            labels_batch
        )

        # =================================================
        # BACKPROP
        # =================================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # =================================================
        # METRICS
        # =================================================

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()

    accuracy = correct / total

    train_losses.append(running_loss)

    train_accuracies.append(accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.4f}"
    )

# =========================================================
# VALIDATION
# =========================================================

print("\nVALIDATION\n")

model.eval()

all_preds = []

all_labels = []

with torch.no_grad():

    for mel, labels_batch in val_loader:

        mel = mel.to(device)

        outputs = model(mel)

        _, predicted = torch.max(
            outputs,
            1
        )

        all_preds.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels_batch.numpy()
        )

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nCONFUSION MATRIX:\n")

print(cm)

# =========================================================
# REPORT
# =========================================================

print("\nCLASSIFICATION REPORT:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_
    )
)

# =========================================================
# PLOT CONFUSION MATRIX
# =========================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

ticks = np.arange(len(labels))

plt.xticks(
    ticks,
    label_encoder.classes_,
    rotation=45
)

plt.yticks(
    ticks,
    label_encoder.classes_
)

plt.xlabel("Predicted")

plt.ylabel("True")

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

# =========================================================
# TRAIN CURVES
# =========================================================

plt.figure(figsize=(10, 5))

plt.plot(train_losses)

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid()

plt.show()

plt.figure(figsize=(10, 5))

plt.plot(train_accuracies)

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid()

plt.show()

joblib.dump(model, 'alpha3.pkl')