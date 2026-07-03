# Music Genre Classification Web App

A production-style **Music Genre Classification** project that combines **audio feature engineering**, **machine learning model selection**, and a **Django web application** for real-time genre prediction from uploaded `.wav` files.

This project was built to demonstrate an end-to-end ML workflow:

- extracting meaningful music/audio features with **librosa**
- training and comparing multiple classical ML models
- improving performance with **hyperparameter tuning** and **feature selection**
- exporting the best model for inference
- integrating the model into a **Django-based upload + prediction app**

---

## Project Highlights

- **End-to-end ML pipeline**: dataset → preprocessing → training → tuning → evaluation → deployment
- **Audio feature extraction** using tempo, beats, spectral features, and **MFCCs**
- **Multiple model comparison**:
  - Support Vector Machine (SVM)
  - Extra Trees Classifier
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- **Feature selection with SelectKBest** to improve classification performance
- **Django web app** for uploading `.wav` files and getting genre predictions
- **Reusable saved model package** that includes model + scaler + feature selector where required

---

## Tech Stack

**Languages & Frameworks**

- Python
- Django

**ML / Data Libraries**

- scikit-learn
- pandas
- numpy
- librosa

**Other Tools**

- Jupyter Notebook
- pickle (model serialization)

---

## Problem Statement

Music files contain rich information such as rhythm, timbre, energy, and frequency patterns. The goal of this project is to automatically classify a song into its **music genre** based on extracted audio features.

The final system allows a user to upload a **`.wav` audio file** through a web interface, after which the application extracts the same features used during training and predicts the most likely genre.

---

## Dataset

The project uses a structured music-genre dataset stored as a CSV file.

### Dataset summary

- **Rows:** 1000
- **Columns:** 30
- **Target column:** `label`
- **Human-readable genre column:** `class_name`

### Core columns

- `tempo`
- `beats`
- `chroma_stft`
- `rmse`
- `spectral_centroid`
- `spectral_bandwidth`
- `rolloff`
- `zero_crossing_rate`
- `mfcc1` to `mfcc20`

Each row represents one audio sample with its extracted numerical features and corresponding genre label.

---

## Features Used

The model is trained on **28 audio features**.

### Rhythm Features

- `tempo`
- `beats`

### Spectral / Energy Features

- `chroma_stft`
- `rmse`
- `spectral_centroid`
- `spectral_bandwidth`
- `rolloff`
- `zero_crossing_rate`

### Cepstral Features

- `mfcc1` to `mfcc20`

These features capture different aspects of an audio signal:

- **Tempo / beats** → rhythm and pacing
- **Spectral centroid / bandwidth / rolloff** → brightness and distribution of frequencies
- **RMSE** → signal energy
- **Zero crossing rate** → noisiness / percussive content
- **MFCCs** → timbral characteristics, often highly useful for music/audio classification

---

## ML Workflow

### 1) Data Preparation

- Load the dataset from CSV
- Select the 28 numerical feature columns
- Use the genre `label` as the target
- Perform a **train-test split** with stratification

### 2) Model Training

The following models were trained and compared:

- **Support Vector Machine**
- **Extra Trees Classifier**
- **Random Forest Classifier**
- **K-Nearest Neighbors**

### 3) Hyperparameter Tuning

Hyperparameter tuning was applied to improve performance of the models.This included tuning parameters such as:

- **SVM**: `C`, `gamma`, `kernel`
- **Random Forest / Extra Trees**: `n_estimators`, `max_depth`, `min_samples_split`, etc.
- **KNN**: `n_neighbors`, `weights`, distance metrics

### 4) Feature Selection

To improve generalization and reduce noise, **SelectKBest** with **ANOVA F-test (`f_classif`)** was applied.

This was especially useful with SVM, resulting in the strongest final setup:

> **Tuned SVM + SelectKBest**

---

## Final Model

The final deployed model is the **best-performing model selected after training and evaluation**.

### Saved model package

The project stores the trained model in a serialized file:

```bash
best_music_genre_model.pkl
```

The saved package can include:

- `model_name`
- `model`
- `scaler`
- `feature_columns`
- `selector` *(when the best model uses SelectKBest)*

This makes inference reproducible because the same preprocessing steps used during training are also applied during prediction.

---

## Project Architecture

### Training side

1. Load dataset
2. Train multiple ML models
3. Tune hyperparameters
4. Evaluate models
5. Select the best model
6. Save the model package as `.pkl`

### Web app side

1. User uploads a `.wav` file in Django
2. `Metadata.py` extracts audio features using **librosa**
3. `predict.py` loads the saved model package
4. Input features are reordered to match the training feature order
5. Required preprocessing is applied:
   - scaling
   - feature selection (if needed)
6. The model predicts the genre
7. Result is displayed in the UI

---

## Django Inference Pipeline

### `Metadata.py`

Responsible for extracting the same features from an uploaded `.wav` file that were used during training.

Extracted features:

- tempo
- beats
- chroma_stft
- rmse
- spectral_centroid
- spectral_bandwidth
- rolloff
- zero_crossing_rate
- mfcc1 → mfcc20

It returns a **dictionary** of feature values so the prediction code can arrange them in the exact training order.

### `predict.py`

Responsible for:

- loading the saved model package
- converting extracted metadata into a DataFrame
- applying the correct preprocessing depending on the model type
- returning the predicted genre

### `views.py`

Handles:

- file upload from the user
- validation of `.wav` input
- feature extraction
- model prediction
- rendering the final predicted genre on the result page

---

## Project Structure

```bash
Music_Genre_Classification/
│
├── manage.py
├── data.csv
├── model_training.ipynb
├── requirements.txt
├── best_music_genre_model.pkl
│
├── predictor/
│   ├── views.py
│   ├── predict.py
│   ├── Metadata.py
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   ├── templates/
│   │   └── music/
│   │       ├── index.html
│   │       └── result.html
│   └── ...
│
└── models/
    └── best_music_genre_model.pkl
```

> The exact folder layout can vary slightly depending on where the trained model file is stored inside the Django project.

---

## How to Run the Project

### 1) Clone the repository

```bash
git clone <your-repository-url>
cd Music_Genre_Classification
```

### 2) Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Make sure the trained model file exists

Place the trained model file in the folder pointed to by your Django `settings.MODELS` path:

```bash
best_music_genre_model.pkl
```

### 5) Run migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 6) Start the Django server

```bash
python manage.py runserver
```

### 7) Open the app in browser

Visit:

```bash
http://127.0.0.1:8000/
```

Upload a `.wav` file and the app will display the predicted genre.

---

## Example Prediction Flow

1. Upload a `.wav` file
2. The app extracts:
   - tempo
   - beats
   - spectral features
   - MFCC features
3. Features are arranged in the same order as training
4. Saved model pipeline is loaded
5. Model predicts the genre label
6. Result is shown on the web page

---

## Why This Project Is Strong for a Portfolio

This project demonstrates practical skills across **multiple parts of the ML lifecycle**:

### Machine Learning

- supervised classification
- model comparison
- hyperparameter tuning
- feature selection
- model evaluation
- cross-validation

### Audio / Signal Processing

- feature extraction from raw audio
- use of MFCCs and spectral features
- bridging raw media input to structured ML features

### Deployment / Application Integration

- serializing trained ML models
- building a web interface with Django
- integrating model inference into a real application workflow
- handling file uploads and inference-time preprocessing

---

## Challenges Solved During Development

Some of the practical issues addressed in this project include:

- aligning **training features** with **prediction-time extracted features**
- ensuring uploaded `.wav` files produce the **same feature schema** as the training dataset
- handling model-specific preprocessing at inference time
- saving not just the model, but also the **scaler / selector / feature order**
- debugging feature mismatches between notebook experiments and Django deployment

---

## Possible Future Improvements

There are several ways this project can be extended:

- Add **top-3 genre probabilities** instead of only one predicted label
- Show extracted feature values and confidence scores in the UI
- Support additional audio formats such as `.mp3`
- Move from hand-crafted features to **deep learning on spectrograms**
- Train CNN / CRNN models on **mel spectrograms**
- Add charts for waveform / spectrogram visualization in the web app
- Containerize the app for deployment

---

## Learning Outcomes

Through this project, the following concepts were applied in practice:

- audio feature extraction using **librosa**
- classical ML model training and evaluation
- feature engineering and selection
- hyperparameter tuning
- model persistence with pickle
- integrating ML inference with a Django application

---

## Conclusion

This project is an end-to-end implementation of **Music Genre Classification using Machine Learning and Django**.

It goes beyond a simple notebook experiment by covering:

- structured feature engineering from audio
- multiple model experiments and tuning
- best-model packaging for inference
- real-world deployment through a web interface

It is a strong example of combining **ML + audio processing + web development** into a single usable application.
