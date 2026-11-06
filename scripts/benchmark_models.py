"""
Comprehensive Machine Learning Benchmark Tournament
Compares 10+ classical ML algorithms and ensemble architectures on the 768-sample Indian Music dataset.
Evaluates using 5-Fold Stratified Cross-Validation and held-out test accuracy.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Models
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(PROJECT_ROOT, "datasets/indian_music_data.csv")

print(f"Loading dataset: {data_path}")
df = pd.read_csv(data_path)

feature_cols = [
    'tempo', 'beats', 'chroma_stft', 'rmse', 'spectral_centroid',
    'spectral_bandwidth', 'rolloff', 'zero_crossing_rate',
    'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7',
    'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13',
    'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20'
]

X = df[feature_cols].copy()
y = df['label'].copy()

# Stratified Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature selection and scaling
selector = SelectKBest(score_func=f_classif, k=24)
X_train_kbest = selector.fit_transform(X_train, y_train)
X_test_kbest = selector.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_kbest)
X_test_scaled = scaler.transform(X_test_kbest)

# Models dictionary
models = {
    '1. Tuned SVM (RBF, C=3.0)': SVC(C=3.0, kernel='rbf', probability=True, random_state=42),
    '2. Tuned SVM (RBF, C=5.0)': SVC(C=5.0, kernel='rbf', probability=True, random_state=42),
    '3. Linear SVM (C=1.0)': SVC(C=1.0, kernel='linear', probability=True, random_state=42),
    '4. Random Forest (200 trees)': RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42),
    '5. Extra Trees (200 trees)': ExtraTreesClassifier(n_estimators=200, max_depth=14, random_state=42),
    '6. HistGradientBoosting': HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42),
    '7. Gradient Boosting (Tree)': GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42),
    '8. MLP Neural Network (128x64)': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=42),
    '9. K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    '10. Logistic Regression': LogisticRegression(max_iter=1000, C=1.5, random_state=42),
    '11. Naive Bayes (Gaussian)': GaussianNB(),
}

# Voting Ensemble: Combines SVM + Extra Trees + MLP
ensemble_estimators = [
    ('svm', SVC(C=3.0, kernel='rbf', probability=True, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=200, max_depth=14, random_state=42)),
    ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42)),
]
models['12. Soft Voting Ensemble'] = VotingClassifier(estimators=ensemble_estimators, voting='soft')

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning Machine Learning Tournament (5-Fold Stratified CV)...")
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results.append({
        'Model': name,
        '5-Fold CV Mean': f"{cv_scores.mean() * 100:.2f}%",
        'CV StdDev': f"±{cv_scores.std() * 100:.2f}%",
        'Test Accuracy': f"{test_acc * 100:.2f}%",
        'Weighted F1': f"{f1 * 100:.2f}%",
        '_acc_val': test_acc
    })
    print(f"  Processed: {name} -> Test Acc: {test_acc*100:.2f}%")

# Sort by test accuracy descending
results.sort(key=lambda x: x['_acc_val'], reverse=True)
for r in results:
    del r['_acc_val']

res_df = pd.DataFrame(results)
print("\n" + "=" * 80)
print("🏆 MACHINE LEARNING MODEL TOURNAMENT LEADERBOARD (8 INDIAN GENRES)")
print("=" * 80)
print(res_df.to_string(index=False))
