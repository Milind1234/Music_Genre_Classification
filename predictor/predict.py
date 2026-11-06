import os
import pickle
import numpy as np
import pandas as pd
from django.conf import settings


def predict_cnn(y, sr=22050, model_filename='indian_music_cnn_model.pt'):
    """Runs 2D CNN inference directly on audio Mel-Spectrogram."""
    import torch
    import torch.nn.functional as F
    import librosa
    from scripts.train_cnn_model import MusicGenreCNN

    path = os.path.join(settings.MODELS, model_filename)
    if not os.path.exists(path):
        return None

    checkpoint = torch.load(path, map_location='cpu')
    genres = checkpoint['genres']
    target_size = checkpoint.get('target_size', (128, 128))

    max_samples = 45 * sr
    if len(y) > max_samples:
        y = y[:max_samples]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    denom = (mel_db.max() - mel_db.min())
    if denom == 0:
        denom = 1e-6
    norm = (mel_db - mel_db.min()) / denom

    tensor = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

    model = MusicGenreCNN(num_classes=len(genres))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        logits = model(resized)
        probs = F.softmax(logits, dim=1).squeeze().numpy()

    pred_idx = int(np.argmax(probs))
    pred_genre = genres[pred_idx]

    order = np.argsort(probs)[::-1]
    all_probabilities = [
        {
            'genre': str(genres[i]).replace('_', ' ').title(),
            'genre_raw': str(genres[i]).lower(),
            'confidence': round(float(probs[i] * 100), 1),
            'rank': rank + 1
        }
        for rank, i in enumerate(order)
    ]
    top3 = all_probabilities[:3]
    top_confidence = top3[0]['confidence'] if top3 else 0.0

    return {
        'predicted_genre': str(pred_genre).lower(),
        'top_confidence': top_confidence,
        'top3': top3,
        'all_probabilities': all_probabilities,
        'model_type': 'indian',
        'model_display_name': f'Indian Deep Learning CNN ({len(genres)} Genres)',
        'architecture': 'cnn'
    }


def predict_gen(meta1, model_type="indian", architecture="svm", y=None, sr=22050):
    """
    Takes audio metadata dictionary and runs inference using the selected model pipeline.
    Supports classical ML (SVM) and deep learning (2D Mel-Spectrogram CNN).
    """
    if architecture == "cnn" and model_type == "indian" and y is not None:
        cnn_result = predict_cnn(y, sr)
        if cnn_result is not None:
            return cnn_result
    if model_type == "indian":
        model_filename = 'indian_music_genre_model.pkl'
        model_display_name = 'Indian Music Classifier'
    else:
        model_filename = 'best_music_genre_model.pkl'
        model_display_name = 'Global / Western Classifier (GTZAN 10 Genres)'

    path = os.path.join(settings.MODELS, model_filename)

    with open(path, 'rb') as pickled:
        model_package = pickle.load(pickled)

    model = model_package["model"]
    feature_columns = model_package["feature_columns"]
    classes = model.classes_

    if model_type == "indian":
        model_display_name = f"Indian Music Classifier ({len(classes)} Genres)"

    # Build DataFrame in exact training feature order
    sample_df = pd.DataFrame([[meta1[col] for col in feature_columns]], columns=feature_columns)

    # Feature transformation pipeline
    if "selector" in model_package and model_package["selector"] is not None:
        selector = model_package["selector"]
        sample_selected = selector.transform(sample_df)
    else:
        sample_selected = sample_df

    if "scaler" in model_package and model_package["scaler"] is not None:
        scaler = model_package["scaler"]
        sample_transformed = scaler.transform(sample_selected)
    else:
        sample_transformed = sample_selected

    pred = model.predict(sample_transformed)[0]

    # Compute probability / confidence distribution across classes
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(sample_transformed)[0]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(sample_transformed)[0]
        # Softmax with numerical stability
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
    else:
        probs = np.zeros(len(model.classes_))
        match_idx = np.where(model.classes_ == pred)[0]
        if len(match_idx):
            probs[match_idx[0]] = 1.0

    classes = model.classes_
    order = np.argsort(probs)[::-1]

    all_probabilities = [
        {
            'genre': str(classes[i]).replace('_', ' ').title(),
            'genre_raw': str(classes[i]).lower(),
            'confidence': round(float(probs[i] * 100), 1),
            'rank': rank + 1
        }
        for rank, i in enumerate(order)
    ]

    top3 = all_probabilities[:3]
    top_confidence = top3[0]['confidence'] if top3 else 0.0

    return {
        'predicted_genre': str(pred).lower(),
        'top_confidence': top_confidence,
        'top3': top3,
        'all_probabilities': all_probabilities,
        'model_type': model_type,
        'model_display_name': model_display_name,
    }
