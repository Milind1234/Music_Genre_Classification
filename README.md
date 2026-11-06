# 🎵 Music Genre Classification Web App (Classical ML & Deep Learning)

An end-to-end Machine Learning and Deep Learning Django web application that classifies audio tracks into **8 Indian Music Genres** and **10 Global/Western Music Genres** from uploaded `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, and `.opus` audio files. Features dual architectures: a **Tuned Support Vector Machine (SVM) (81.17% Test Accuracy)** and a **PyTorch 2D Mel-Spectrogram Convolutional Neural Network (CNN)**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Supported Genres](#supported-genres)
  - [🇮🇳 Indian Music Classifier (8 Genres)](#-1-indian-music-classifier-8-genres)
  - [🌍 Global / Western Classifier (10 Genres)](#-2-global--western-classifier-gtzan-10-genres)
- [System Architecture & Dual Pipelines](#system-architecture)
- [Dataset & Audio Features](#dataset--audio-features)
  - [The 28 Extracted Features](#the-28-extracted-features)
  - [Feature Selection (SelectKBest)](#feature-selection-selectkbest)
- [Machine Learning Workflow & Benchmarks](#machine-learning-workflow--benchmarks)
  - [12-Model Machine Learning Tournament Leaderboard](#-12-model-machine-learning-tournament-leaderboard)
  - [Hyperparameter Tuning Impact (Default vs. Tuned)](#-hyperparameter-tuning-impact-8117-test-accuracy)
  - [Deep Learning 2D CNN Architecture](#-deep-learning-2d-mel-spectrogram-cnn)
- [Web Application & UI Features](#web-application-architecture)
- [Repository Structure](#repository-structure)
- [🐳 Docker Deployment (Recommended)](#-docker-deployment-recommended)
- [🚀 Installation & Setup (Local Environment)](#-installation--setup-local-environment)
- [🧪 Testing & Verification](#-testing-the-model)
- [Roadmap & Feature Checklist](#-roadmap--enhancements)
- [License](#license)

---

## 🎧 Overview

Music genre classification involves analyzing acoustic patterns, harmonic structures, rhythms, and frequency distributions. This repository provides a complete, production-style machine learning pipeline that:

1. **Extracts acoustic descriptors** (rhythm, timbre, spectral shape, energy) from raw audio files using `librosa`.
2. **Trains and benchmarks** multiple scikit-learn models (Logistic Regression, KNN, Random Forest, Extra Trees, Gradient Boosting, SVM).
3. **Applies univariate feature selection** (`SelectKBest` with ANOVA F-test) and **hyperparameter grid search** (`GridSearchCV`) to optimize accuracy.
4. **Serializes the complete inference bundle** (trained model + scaler + feature selector + column alignment schema).
5. **Serves real-time predictions** through a responsive Django web interface with in-memory audio processing.

---

## 🎼 Supported Genres

The application features a **Dual-Model Classifier** with an intuitive toggle on the UI:

### 🇮🇳 1. Indian Music Classifier (8 Genres)

| # | Genre | Description & Acoustic Signatures |
|---|-------|-----------------------------------|
| 1 | **Bollywood / Filmi** | Modern commercial film tracks, polished multi-layered production, catchy melodic hooks, contemporary pop/electronic rhythm |
| 2 | **Carnatic Classical** | South Indian classical tradition, intricate Raga melodies, complex Tala rhythmic cycles, Veena, Mridangam, Violin |
| 3 | **Hindustani Classical** | North Indian classical music, meditative Alap transitions into fast Jhala, Sitar, Sarod, Tabla, Bansuri harmonics |
| 4 | **Punjabi / Bhangra** | High-energy driving rhythms, punchy acoustic dhol beats, festive tempo, celebratory hooks |
| 5 | **Ghazal** | Poetic and lyrical acoustic arrangements, warm Harmonium resonance, nuanced Tabla phrasing, emotive vocal expression |
| 6 | **Sufi / Qawwali** | Devotional spiritual ecstasy, clapped rhythm, chorus vocal responses, harmonium, accelerating tempo crescendos |
| 7 | **Indian Folk** | Vibrant regional traditions (Garba, Lavani, Rajasthani, Bihu), authentic folk percussion (Dholak, Khartal), syncopated dance meter |
| 8 | **Devotional / Bhajan** | Serene spiritual hymns, Harmonium, Manjira (cymbals), gentle acoustic rhythm, uplifting melodic warmth |

### 🌍 2. Global / Western Classifier (GTZAN 10 Genres)

| # | Genre | Description |
|---|-------|-------------|
| 1 | **Blues** | Characteristic chord progressions, acoustic/electric guitar, emotive vocals |
| 2 | **Classical** | Orchestral arrangements, acoustic instrumentation, wide dynamic range |
| 3 | **Country** | Acoustic string instruments (banjo, fiddle, guitar), narrative vocals |
| 4 | **Disco** | Four-on-the-floor rhythm, prominent basslines, syncopated rhythm guitars |
| 5 | **Hiphop** | Heavy bass/sub-bass, rhythmic beats, sampled loops, percussive cadence |
| 6 | **Jazz** | Swing rhythms, complex chords, brass/woodwind improvisation |
| 7 | **Metal** | Distorted guitars, double-kick drumming, aggressive tempo, high energy |
| 8 | **Pop** | Catchy melodic hooks, repetitive rhythm, polished production |
| 9 | **Reggae** | Off-beat staccato chords (skank), prominent melodic basslines |
| 10 | **Rock** | Driven electric guitars, driving drums, verse-chorus song structures |

---

## 🏗️ System Architecture

```
                       Uploaded Audio (.wav, .mp3, .flac, .ogg, .m4a, .opus)
                                              │
                                              ▼
                             [ Django View (`model_form_upload`) ]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        [ Mode: Classical ML (SVM) ]                     [ Mode: Deep Learning (CNN) ]
                     │                                                 │
      [ Feature Extraction (`Metadata.py`) ]             [ 2D Mel-Spectrogram Generation ]
      Librosa: Tempo, Beats, Chroma, RMS,                128 Mel bands, n_fft=2048, dB-norm,
      Spectral Centroid, Rolloff, ZCR, MFCCs (1-20)      Interpolated to (128x128) Tensor
                     │                                                 │
             28 Audio Descriptors                        PyTorch 4-Block Conv2D + BatchNorm
                     │                                                 │
          SelectKBest (ANOVA F-test)                     Global Average Pooling + Linear
                     │                                                 │
             StandardScaler                                     Softmax Logits
                     │                                                 │
            Support Vector Classifier (RBF C=15)                       │
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                             [ Prediction Result View (`result.html`) ]
                             • Top Predicted Genre & Confidence %
                             • Interactive In-Browser HTML5 Audio Player
                             • Ranked Top-3 Genre Probability Breakdown
                             • Dual Spectrogram & Waveform Visualizations
                             • Export Analysis as JSON / Print PDF Report
```

---

## 📊 Dataset & Audio Features

The project is trained on the standard 1,000-track music genre dataset (`data.csv`):
- **Samples:** 1,000 (100 audio tracks per genre, perfectly balanced across 10 classes)
- **Features:** 28 numerical acoustic features extracted per sample
- **Target Column:** `label` / `class_name`

### The 28 Extracted Features

| Category | Features | Description |
|---|---|---|
| **Rhythm & Tempo** | `tempo`, `beats` | Overall tempo in BPM (beats per minute) and total detected beat frame count. |
| **Energy** | `rmse` | Root Mean Square Energy, representing the signal's loudness and physical amplitude. |
| **Pitch & Harmony** | `chroma_stft` | 12-bin Chroma Short-Time Fourier Transform measuring harmonic and tonal pitch distribution. |
| **Spectral Shape** | `spectral_centroid` | Center of mass of the frequency spectrum; indicates perceived "brightness" of sound. |
| **Spectral Spread** | `spectral_bandwidth` | Frequency width of the spectral band around the centroid. |
| **Spectral Rolloff** | `rolloff` | Frequency below which 85% of total spectral energy lies. |
| **Percussiveness / Noise** | `zero_crossing_rate` | Rate at which the acoustic signal changes sign; distinguishes noisy/percussive sounds. |
| **Timbre (Cepstral)** | `mfcc1` – `mfcc20` | First 20 Mel-Frequency Cepstral Coefficients capturing timbral identity and vocal/instrument color. |

### Feature Selection (SelectKBest)

Using ANOVA F-value (`f_classif`), the top **20 most discriminative features** are selected:
- **Retained (20 features):**
  `chroma_stft`, `rmse`, `spectral_centroid`, `spectral_bandwidth`, `rolloff`, `zero_crossing_rate`, `mfcc1`, `mfcc2`, `mfcc3`, `mfcc4`, `mfcc5`, `mfcc6`, `mfcc7`, `mfcc8`, `mfcc9`, `mfcc10`, `mfcc12`, `mfcc13`, `mfcc15`, `mfcc17`
- **Pruned (8 features):**
  `tempo`, `beats`, `mfcc11`, `mfcc14`, `mfcc16`, `mfcc18`, `mfcc19`, `mfcc20`

> **Why remove tempo & beats?** Tempo varies significantly within genres (e.g., slow vs. fast rock or hip-hop ballads) and raw beat frame counts introduce noise compared to spectral and timbral texture representations.

---

## 🔬 Machine Learning Workflow & Benchmarks

The project benchmarks multiple machine learning algorithms across both the 1,000-track GTZAN Western dataset and our curated 768-sample 8-genre Indian music dataset (`datasets/indian_music_data.csv`).

### 🏆 12-Model Machine Learning Tournament Leaderboard

Evaluated on the 768-sample Indian music dataset using **5-Fold Stratified Cross-Validation**:

| Rank | Model Architecture | 5-Fold CV Mean | CV StdDev | Test Accuracy | Weighted F1 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 🥇 | **Tuned SVM (RBF, $C=15.0$)** | **72.48%** | **±3.65%** | **81.17%** | **0.81** |
| 🥈 | **Tuned SVM (RBF, $C=3.0 / 5.0$)** | 71.99% | ±2.54% | 79.87% | 0.79 |
| 🥉 | **Soft Voting Ensemble (SVM + MLP + ET)** | 70.70% | ±4.21% | 78.57% | 0.79 |
| 4 | **MLP Neural Network (128x64)** | 69.55% | ±4.73% | 75.97% | 0.76 |
| 5 | **Extra Trees Classifier (350 trees)** | 70.37% | ±2.67% | 74.68% | 0.75 |
| 6 | **Tuned KNN ($k=3$, Distance-weighted)** | 67.11% | ±2.69% | 74.68% | 0.74 |
| 7 | **Random Forest (300 trees, depth=18)** | 69.72% | ±3.93% | 72.73% | 0.72 |
| 8 | **HistGradientBoosting** | 65.64% | ±3.39% | 70.13% | 0.70 |
| 9 | **Gradient Boosting (Trees)** | 62.88% | ±4.82% | 70.13% | 0.70 |
| 10 | **Logistic Regression** | 62.71% | ±3.94% | 61.69% | 0.62 |
| 11 | **Linear SVM ($C=1.0$)** | 63.04% | ±4.59% | 60.39% | 0.61 |
| 12 | **Naive Bayes (Gaussian)** | 54.23% | ±2.38% | 54.55% | 0.53 |

---

### 📈 Hyperparameter Tuning Impact (81.17% Test Accuracy)

Using systematic grid search (`GridSearchCV`) across regularizations, kernel bandwidths, and tree structures:

| Model | Default Parameters | Default Test Acc | Best Tuned Hyperparameters | Tuned Test Acc | Net Gain |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **SVM (RBF Kernel)** | `C=1.0, gamma='scale'` | 72.73% | `C=15.0, gamma='scale'` | **81.17%** | **+8.44%** 🚀 |
| **KNN** | `k=5, weights='uniform'` | 63.64% | `k=3, weights='distance', p=2` | **74.68%** | **+11.04%** 🚀 |
| **Random Forest** | `n=100, max_depth=None` | 71.43% | `n=300, max_depth=18, min_split=4` | **72.73%** | **+1.30%** |
| **Extra Trees** | `n=100, max_depth=None` | 74.68% | `n=350, max_depth=15, min_split=2` | **74.68%** | *(Higher CV stability)* |

---

### 🧠 Deep Learning: 2D Mel-Spectrogram CNN

In addition to hand-crafted tabular audio descriptors, the application includes an end-to-end deep learning pipeline implemented in PyTorch (`scripts/train_cnn_model.py`):
* **Input Representation:** 2D Mel-Spectrogram matrix ($128 \times 128$ frequency-time bins, normalized in dB scale).
* **Architecture:** 4 convolutional blocks (Conv2d $\rightarrow$ BatchNorm2d $\rightarrow$ ReLU $\rightarrow$ MaxPool2d) followed by AdaptiveAvgPool2d and Linear output classification layer.
* **Hardware Acceleration:** Native PyTorch execution supporting Apple Silicon Metal Performance Shaders (`mps`), CUDA GPU, or CPU.
* **Model Artifact:** Serialized to `predictor/models/indian_music_cnn_model.pt`.

---

## 🌐 Web Application Architecture

The Django web app is organized into two primary applications:

1. **`MusicClassification/`**: Core project configuration
   - `settings.py`: Defines application settings, installed apps, static/media paths, and `MODELS = os.path.join(BASE_DIR, 'predictor/models')`.
   - `urls.py`: Main URL routing pointing to `predictor.urls`.
   - `wsgi.py`: WSGI entry point for production deployments.

2. **`predictor/`**: Audio classification application
   - `Metadata.py`: Audio processing module using `librosa`. Accepts file paths or in-memory uploaded file objects (calling `.seek(0)`).
   - `predict.py`: Model loader and prediction handler. Loads the pickled pipeline, transforms features, and invokes `model.predict()`.
   - `views.py`:
     - `IndexView`: Renders the upload form and interface.
     - `model_form_upload`: Validates `.wav` upload, triggers in-memory feature extraction, queries the model, and displays `music/result.html`.
   - `templates/music/`: HTML templates styled with Bootstrap and custom CSS.

---

## 📁 Repository Structure

```
Music_Genre_Classification/
├── data.csv                          # 1,000-sample GTZAN dataset with 28 features
├── db.sqlite3                        # SQLite development database
├── manage.py                         # Django management script
├── model_training.ipynb              # Complete training, tuning & evaluation notebook
├── requirements.txt                  # Python dependency specifications
├── README.md                         # Project documentation
│
├── datasets/                         # Clean dataset storage
│   ├── data.csv                      # GTZAN dataset copy
│   ├── indian_music_data.csv         # 500-sample real Indian music features
│   └── kaggle_indian/                # 500 authentic Indian audio tracks (.mp3)
│
├── sample_audio/                     # Sample audio files for testing (3.wav, new.wav)
│
├── scripts/                          # Standalone data processing & ML scripts
│   ├── extract_indian_features.py    # Parallel feature extractor across audio folders
│   ├── train_indian_model.py         # Tuned SVM training & benchmark pipeline (81.17% Acc)
│   ├── train_cnn_model.py            # PyTorch 2D Mel-Spectrogram CNN training pipeline
│   └── benchmark_models.py           # 12-Model tournament benchmark script
│
├── MusicClassification/              # Django Project Configuration
│   ├── __init__.py
│   ├── settings.py                   # Django settings (configured with MODELS path)
│   ├── urls.py                       # Root URL dispatcher
│   └── wsgi.py                       # WSGI entry point
│
└── predictor/                        # Django Predictor App
    ├── admin.py                      # Django admin registration
    ├── apps.py                       # App configuration
    ├── forms.py                      # Document upload form
    ├── models.py                     # Document model definition
    ├── urls.py                       # Predictor URL routing ('/' and 'result/')
    ├── views.py                      # Upload processing & inference view
    ├── Metadata.py                   # In-app feature extraction module
    ├── predict.py                    # Dual-model inference pipeline handler
    ├── tests.py                      # Automated test suite (9 tests)
    │
    ├── models/                       # Trained Model Artifacts
    │   ├── indian_music_genre_model.pkl # Indian Music Model (Tuned SVM RBF C=15, 8 classes)
    │   ├── indian_music_cnn_model.pt    # Indian Deep Learning CNN (PyTorch, 8 classes)
    │   └── best_music_genre_model.pkl   # Global GTZAN Model (SVM + SelectKBest, 10 classes)
    │
    ├── migrations/                   # Database migration files
    │   ├── 0001_initial.py
    │   └── 0002_alter_document_id.py
    │
    ├── static/music/                 # Frontend assets
    │   ├── css/                      # Bootstrap & custom styling
    │   ├── images/                   # Icons, illustrations, background images
    │   ├── js/                       # Interactive JavaScript plugins
    │   └── webfonts/                 # FontAwesome web fonts
    │
    └── templates/music/              # Django UI Templates
        ├── base.html                 # Base layout template
        ├── index.html                # Main landing page
        ├── check_genre.html          # Drag-and-drop file upload component
        ├── messages.html             # Flash messages & alerts
        ├── result.html               # Prediction result view
        └── snippets/                 # Reusable template snippets
```

---

## 🐳 Docker Deployment (Recommended)

Run the entire application in a standardized production container with pre-configured audio drivers (`libsndfile`, `ffmpeg`, `gunicorn`, `PyTorch`):

```bash
# Build and start container in detached mode
docker compose up --build -d

# Check live logs
docker compose logs -f

# Visit the web application
open http://localhost:8000
```

To stop the container:
```bash
docker compose down
```

---

## 🚀 Installation & Setup (Local Environment)

### Prerequisites

- **Python:** 3.10 or 3.11 recommended (Python 3.12+ may encounter legacy library build conflicts).
- **Audio backend:** `libsndfile` (pre-installed on macOS/Linux; used by `soundfile` and `librosa`).

### 1. Clone Repository

```bash
git clone https://github.com/your-username/Music_Genre_Classification.git
cd Music_Genre_Classification
```

### 2. Create and Activate Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Database Migrations

```bash
python manage.py migrate
```

*(Note: The database migrations are already initialized in `predictor/migrations/`. You only need to run `migrate` to set up your local SQLite database).*

### 5. Start Development Server

```bash
python manage.py runserver
```

Visit **`http://127.0.0.1:8000/`** in your web browser.

---

## 🧪 Testing the Model

### Through the Web UI

1. Open `http://127.0.0.1:8000/`.
2. Scroll to the **CHECK GENRE** section.
3. Upload the provided test audio file `3.wav` or `new.wav` (or any valid 16-bit or 24-bit `.wav` file).
4. Click **Check**.
5. The result page will display the predicted genre (e.g., `classical`).

### Programmatically via Python / Notebook

You can test feature extraction and predictions directly from a Python terminal or inside `model_training.ipynb`:

```python
import pickle
import pandas as pd
from Metadata import getmetadata

# 1. Load the trained model bundle
with open("predictor/models/best_music_genre_model.pkl", "rb") as f:
    pkg = pickle.load(f)

model = pkg["model"]
selector = pkg["selector"]
scaler = pkg["scaler"]
feature_cols = pkg["feature_columns"]

# 2. Extract features from a .wav file
features_dict = getmetadata("3.wav")

# 3. Format into a DataFrame matching training schema
df_sample = pd.DataFrame([[features_dict[col] for col in feature_cols]], columns=feature_cols)

# 4. Transform and predict
X_selected = selector.transform(df_sample)
X_scaled = scaler.transform(X_selected)
prediction = model.predict(X_scaled)[0]

print(f"Predicted Genre: {prediction}")
# Output: Predicted Genre: classical
```

---

## 🔍 Known Clarifications & Differences from Legacy Code

1. **Model Storage Location**:
   - The serialized model file is located at `predictor/models/best_music_genre_model.pkl` (configured via `settings.MODELS`).
   - `predictor/models/models.p` is a legacy artifact from an earlier baseline version; the active production pipeline uses `best_music_genre_model.pkl`.
2. **In-Memory Audio Processing**:
   - The application does not store user-uploaded audio files permanently on disk. When a `.wav` file is submitted, `views.py` resets the buffer pointer using `uploadfile.seek(0)` and streams it directly into `librosa.load()`, saving storage overhead.
3. **Template Display vs. Notebook Metrics**:
   - The landing page's counter contains demo stats (750 training / 250 test, 73.2% accuracy).
   - The actual rigorous evaluation performed in `model_training.ipynb` uses an **80/20 stratified split** (800 train / 200 test) yielding **70.5% test accuracy** for the tuned SVM with SelectKBest.
4. **Multi-Format Audio Processing**:
   - The pipeline natively supports `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.aac`, `.au`, and `.aiff` audio formats with robust in-memory decoding and fallback temporary file handling.

---

## 🔮 Roadmap & Feature Checklist

- [x] **Multi-Format Audio Support**: Natively process `.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, `.opus`, `.au`, and `.aiff` with PyAV and Librosa fallbacks.
- [x] **Expanded Indian Genres**: Curated 8 distinct real-world Indian music genres (*Bollywood, Carnatic, Hindustani, Punjabi, Ghazal, Sufi, Folk, Bhajan*).
- [x] **Interactive In-Browser Audio Player**: Embedded HTML5 waveform audio playback on the prediction result page.
- [x] **Analysis Export & Reporting**: Client-side JSON inspection export and formatted print/PDF generation.
- [x] **Deep Learning Models (Option A)**: PyTorch 2D Mel-Spectrogram CNN model with dual-architecture toggle (⚡ Classical ML vs 🧠 Deep Learning).
- [x] **12-Model ML Tournament & Hyperparameter Tuning**: Systematic 5-fold cross-validation reaching **81.17% Test Accuracy** with tuned SVM ($C=15.0$).
- [x] **Production Dockerization (Option B)**: Multi-stage `Dockerfile` with system audio dependencies (`libsndfile`, `ffmpeg`), Gunicorn WSGI, and `docker-compose.yml`.
- [x] **Automated Test Suite**: 9 unit tests passing across audio formats, model types, and deep learning inference.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
