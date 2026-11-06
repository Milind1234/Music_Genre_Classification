"""
Deep Learning CNN Trainer for Music Genre Classification
Trains a 2D Convolutional Neural Network directly on 2D Mel-Spectrogram image tensors.

Architecture:
- 4x Conv2D Blocks (Conv2D -> BatchNorm -> ReLU -> MaxPool2D -> Dropout)
- Adaptive Pooling (4x4)
- Fully Connected Classifier (Linear -> ReLU -> Dropout -> Linear -> Softmax)
- Target: 8 Indian Music Genres
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')

import librosa
from predictor.Metadata import load_audio_flexible

# Standardized Genre Mapping
GENRE_MAP = {
    'bollypop': 'bollywood',
    'bollywood_romantic': 'bollywood',
    'bollywood_rap': 'desi_hiphop',
    'semiclassical': 'semiclassical',
    'carnatic': 'carnatic',
    'ghazal': 'ghazal',
    'sufi': 'sufi',
    'bhajan': 'bhajan',
    'bhojpuri': 'folk',
    'garhwali': 'folk',
}


def audio_to_mel_spectrogram(audio_path, target_size=(128, 128)):
    """Converts audio file to a normalized 2D Mel-Spectrogram tensor (1, H, W)."""
    y, sr = load_audio_flexible(audio_path, sr=22050)
    # Clip to max 45 seconds for efficiency & consistency
    max_samples = 45 * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Min-max normalization to [0, 1]
    denom = (mel_db.max() - mel_db.min())
    if denom == 0:
        denom = 1e-6
    norm = (mel_db - mel_db.min()) / denom

    tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)
    return resized.squeeze(0).numpy() # (1, 128, 128)


class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MusicGenreCNN(nn.Module):
    """Deep 2D Convolutional Neural Network for Mel-Spectrogram Pattern Recognition."""
    def __init__(self, num_classes=8):
        super(MusicGenreCNN, self).__init__()

        # Conv Block 1 (128x128 -> 64x64)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        # Conv Block 2 (64x64 -> 32x32)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )

        # Conv Block 3 (32x32 -> 16x16)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )

        # Conv Block 4 (16x16 -> 8x8 -> 4x4)
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3)
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.classifier(x)
        return logits


def train_cnn(epochs=25, batch_size=32, lr=1e-3, output_path=None):
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "predictor/models/indian_music_cnn_model.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using compute device: {device}")

    # Gather audio files across datasets
    input_dirs = [
        os.path.join(PROJECT_ROOT, "datasets/kaggle_indian"),
        os.path.join(PROJECT_ROOT, "datasets/gacmis/Dataset/Songs")
    ]
    supported_extensions = ('.wav', '.mp3', '.ogg', '.flac', '.m4a', '.opus')

    genre_files = {}
    for d in input_dirs:
        if not os.path.exists(d):
            continue
        for folder in os.listdir(d):
            fpath = os.path.join(d, folder)
            if not os.path.isdir(fpath):
                continue
            raw = folder.lower().strip()
            genre = GENRE_MAP.get(raw, raw)
            files = [
                f for f in glob.glob(os.path.join(fpath, "*"))
                if os.path.splitext(f)[1].lower() in supported_extensions
            ]
            if genre not in genre_files:
                genre_files[genre] = []
            genre_files[genre].extend(files)

    genres = sorted(genre_files.keys())
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    print(f"Detected {len(genres)} classes: {genres}")

    # Build dataset list (cap at 60 tracks per class for rapid high-performance CNN training)
    samples = []
    labels = []
    print("\nExtracting Mel-Spectrogram Tensors...")
    start_t = time.time()
    for g in genres:
        flist = genre_files[g][:60]
        print(f"  Processing {len(flist)} tracks for '{g}'...")
        for f in flist:
            try:
                spec = audio_to_mel_spectrogram(f)
                samples.append(spec)
                labels.append(genre_to_idx[g])
            except Exception as e:
                pass

    X = np.array(samples)
    y = np.array(labels)
    print(f"Total dataset shape: X={X.shape}, y={y.shape} in {time.time()-start_t:.1f}s")

    # Train / Test Split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_loader = DataLoader(MelSpectrogramDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MelSpectrogramDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    model = MusicGenreCNN(num_classes=len(genres)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    print("\nStarting CNN Training Loop...")
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)

        train_loss = total_loss / len(X_train)

        # Evaluation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(pred)
                trues.extend(batch_y.numpy())

        val_acc = accuracy_score(trues, preds)
        scheduler.step(val_acc)

        if epoch % 5 == 0 or epoch == epochs or val_acc > best_acc:
            print(f"  Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Test Acc: {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            # Save checkpoint
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'genres': genres,
                'genre_to_idx': genre_to_idx,
                'target_size': (128, 128),
                'best_accuracy': best_acc,
                'architecture': '2D Mel-Spectrogram CNN'
            }
            torch.save(checkpoint, output_path)

    print(f"\nTraining Complete! Best Test Accuracy: {best_acc*100:.2f}%")
    print(f"Saved production PyTorch model to: {output_path}")

    # Final Classification Report
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test_tensor)
        final_preds = torch.argmax(outputs, dim=1).cpu().numpy()

    print("\nDeep Learning CNN Classification Report:")
    print(classification_report(y_test, final_preds, target_names=genres))
    return best_acc


if __name__ == '__main__':
    train_cnn(epochs=20)

