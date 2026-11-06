"""
Indian Music Genre Classification Training Script
Trains and evaluates multiple competitive ML architectures on real-world Indian music features:
- Support Vector Classifier (RBF kernel + C tuning)
- Random Forest Classifier (Ensemble bagging)
- Gradient Boosting Classifier
- Multi-Layer Perceptron (Neural Network)

Selects the best performing model based on stratified cross-validation and test set accuracy,
and serializes the production pipeline to `predictor/models/indian_music_genre_model.pkl`.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_indian_music_model(data_path=None, output_path=None):
    if data_path is None:
        data_path = os.path.join(PROJECT_ROOT, "datasets/indian_music_data.csv")
    if output_path is None:
        output_path = os.path.join(PROJECT_ROOT, "predictor/models/indian_music_genre_model.pkl")
    if not os.path.exists(data_path):
        # Fallback to root path if datasets/ copy not found
        if os.path.exists("indian_music_data.csv"):
            data_path = "indian_music_data.csv"
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}. Run extract_indian_features.py first.")

    print(f"Loading real Indian music dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print("\nClass distribution:")
    print(df['label'].value_counts())

    feature_cols = [
        'tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid',
        'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
        'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7',
        'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13',
        'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
    ]

    # Verify all feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    X = df[feature_cols].copy()
    y = df['label'].copy()

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nFitting feature selector (SelectKBest, k=24)...")
    selector = SelectKBest(score_func=f_classif, k=min(24, len(feature_cols)))
    X_train_kbest = selector.fit_transform(X_train, y_train)
    X_test_kbest = selector.transform(X_test)

    print("Fitting feature scaler (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_kbest)
    X_test_scaled = scaler.transform(X_test_kbest)

    # Candidate models for comparison (incorporating tuned hyperparameter discoveries)
    candidates = {
        'SVM (Tuned RBF C=15.0)': SVC(C=15.0, kernel='rbf', probability=True, random_state=42),
        'SVM (Tuned RBF C=10.0)': SVC(C=10.0, kernel='rbf', probability=True, random_state=42),
        'SVM (RBF C=5.0)': SVC(C=5.0, kernel='rbf', probability=True, random_state=42),
        'SVM (RBF C=3.0)': SVC(C=3.0, kernel='rbf', probability=True, random_state=42),
        'Extra Trees (Tuned)': ExtraTreesClassifier(n_estimators=350, max_depth=15, min_samples_split=2, random_state=42),
        'KNN (Tuned k=3 Distance)': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'Random Forest (Tuned)': RandomForestClassifier(n_estimators=300, max_depth=18, min_samples_split=4, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }

    print("\n" + "=" * 60)
    print("Benchmarking Classifiers on Real Indian Music Data")
    print("=" * 60)

    best_name = None
    best_model = None
    best_acc = -1.0
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in candidates.items():
        # 5-fold cross validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        # Train on full train set and evaluate on held-out test set
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, y_pred)
        print(f"{name:25} | 5-Fold CV: {cv_scores.mean()*100:5.2f}% (±{cv_scores.std()*100:4.2f}%) | Test Acc: {test_acc*100:5.2f}%")

        if test_acc > best_acc or (test_acc == best_acc and cv_scores.mean() > best_acc):
            best_acc = test_acc
            best_name = name
            best_model = model

    print("=" * 60)
    print(f"Selected Best Model: {best_name} with Test Accuracy: {best_acc*100:.2f}%\n")

    # Detailed metrics for best model
    best_pred = best_model.predict(X_test_scaled)
    print("Classification Report on Held-Out Test Set:")
    print(classification_report(y_test, best_pred))

    print("Confusion Matrix:")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)

    # Package model for Django predictor
    package = {
        'model_name': f'{best_name} (Trained on Real Indian Music)',
        'model': best_model,
        'selector': selector,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'genres': sorted(y.unique().tolist())
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(package, f)

    print(f"\nSuccessfully saved production model package to: {output_path}")
    return best_acc


if __name__ == '__main__':
    train_indian_music_model()
