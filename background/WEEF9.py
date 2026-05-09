import os
import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# =========================================================
# CONFIGURATION
# =========================================================

training_path = r"C:\Users\ADMIN\Documents\BreathIA\balanced_dataset"

validation_path = r"C:\Users\ADMIN\Documents\BreathIA\Validation_Segments_wav"

sr = 22050

n_mels = 128

batch_size = 16

epochs = 200

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")

# =========================================================
# LABELS
# =========================================================

labels = ["Crackle", "Normal", "Wheeze"]

label_encoder = LabelEncoder()

label_encoder.fit(labels)

num_classes = len(labels)

# =========================================================
# MEL TRANSFORM
# =========================================================

mel_transform = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=2048,
    hop_length=512,
    n_mels=n_mels
)

db_transform = T.AmplitudeToDB()

# =========================================================
# DATASET
# =========================================================

class BreathDataset(Dataset):

    def __init__(self, folder):

        self.folder = folder

        self.files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ]

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        filename = self.files[idx]

        filepath = os.path.join(self.folder, filename)

        # =============================================
        # LOAD AUDIO
        # =============================================

        waveform, original_sr = torchaudio.load(filepath)

        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(
                waveform,
                dim=0,
                keepdim=True
            )

        # =============================================
        # RESAMPLE
        # =============================================

        if original_sr != sr:

            resampler = T.Resample(
                orig_freq=original_sr,
                new_freq=sr
            )

            waveform = resampler(waveform)

        # =============================================
        # FIX LENGTH TO 3 SECONDS
        # =============================================

        target_length = sr * 3

        if waveform.shape[1] < target_length:

            pad_size = target_length - waveform.shape[1]

            waveform = torch.nn.functional.pad(
                waveform,
                (0, pad_size)
            )

        else:

            waveform = waveform[:, :target_length]

        # =============================================
        # MEL SPECTROGRAM
        # =============================================

        mel = mel_transform(waveform)

        mel_db = db_transform(mel)

        # =============================================
        # NORMALIZE
        # =============================================

        mel_db = (mel_db - mel_db.mean()) / (
            mel_db.std() + 1e-8
        )

        # =============================================
        # LABEL
        # =============================================

        label_name = filename.replace(
            ".wav",
            ""
        ).split("_")[-1]

        label = label_encoder.transform(
            [label_name]
        )[0]

        return mel_db, label

# =========================================================
# LOADERS
# =========================================================

train_dataset = BreathDataset(training_path)

val_dataset = BreathDataset(validation_path)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)

# =========================================================
# SIMPLE CNN
# =========================================================
class SimpleAudioCNN(nn.Module):

    def __init__(self):

        super(SimpleAudioCNN, self).__init__()

        # =================================================
        # FEATURE EXTRACTOR
        # =================================================

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=(3, 3),
                stride=1,
                padding=0
            ),

            nn.BatchNorm2d(10),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.MaxPool2d(
                kernel_size=(5, 5),
                stride=(5, 5)
            )
        )

        # =================================================
        # AUTOMATIC FLATTEN SIZE
        # =================================================

        dummy = torch.zeros(1, 1, 128, 130)

        dummy_output = self.features(dummy)

        flatten_size = dummy_output.reshape(1, -1).shape[1]

        print(f"Flatten size: {flatten_size}")

        # =================================================
        # CLASSIFIER
        # =================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                flatten_size,
                100
            ),

            nn.ReLU(),

            nn.Linear(
                100,
                num_classes
            )
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x

# =========================================================
# CHECK MEL SHAPE
# =========================================================

sample_mel, _ = train_dataset[0]

print(f"\nSample mel shape: {sample_mel.shape}")

# =========================================================
# CREATE MODEL
# =========================================================

model = SimpleAudioCNN().to(device)

print(model)

# =========================================================
# LOSS AND OPTIMIZER
# =========================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================================================
# TRAIN
# =========================================================

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for mel, labels_batch in train_loader:

        mel = mel.to(device)

        labels_batch = labels_batch.to(device)

        # =============================================
        # FORWARD
        # =============================================

        outputs = model(mel)

        loss = criterion(outputs, labels_batch)

        # =============================================
        # BACKPROP
        # =============================================

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # =============================================
        # METRICS
        # =============================================

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()

    accuracy = correct / total

    print(
        f"\nEpoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.4f}"
    )
# =========================================================
# VALIDATION
# =========================================================

model.eval()

all_preds = []

all_labels = []

with torch.no_grad():

    for mel, labels_batch in val_loader:

        mel = mel.to(device)

        outputs = model(mel)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())

        all_labels.extend(labels_batch.numpy())

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    all_labels,
    all_preds
)

print("\nConfusion Matrix:")
print(cm)

# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=label_encoder.classes_
    )
)

# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.colorbar()

tick_marks = np.arange(len(labels))

plt.xticks(
    tick_marks,
    label_encoder.classes_,
    rotation=45
)

plt.yticks(
    tick_marks,
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