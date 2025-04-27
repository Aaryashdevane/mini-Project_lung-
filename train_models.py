import os
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import joblib

from data_preprocessing import load_and_split_data

# Define paths (update these based on your setup)
audio_dir = '/home/aaryash/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/versions/2/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
metadata_file = '/home/aaryash/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/versions/2/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'

# Ensure output directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Load dataset
X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, label_encoder = load_and_split_data(audio_dir, metadata_file)

def train_and_evaluate_models():
    results = {}

    # SVM
    svm_model = SVC(kernel='linear', probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    results['SVM'] = accuracy_score(y_test, svm_preds)
    joblib.dump(svm_model, 'models/svm_model.pkl')
    print(f"SVM Accuracy: {results['SVM']:.2f}")

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, rf_preds)
    joblib.dump(rf_model, 'models/random_forest_model.pkl')
    print(f"Random Forest Accuracy: {results['Random Forest']:.2f}")

    # ANN
    ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    ann_model.fit(X_train, y_train)
    ann_preds = ann_model.predict(X_test)
    results['ANN'] = accuracy_score(y_test, ann_preds)
    joblib.dump(ann_model, 'models/ann_model.pkl')
    print(f"ANN Accuracy: {results['ANN']:.2f}")

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_preds = knn_model.predict(X_test)
    results['KNN'] = accuracy_score(y_test, knn_preds)
    joblib.dump(knn_model, 'models/knn_model.pkl')
    print(f"KNN Accuracy: {results['KNN']:.2f}")

    # GMM
    gmm_model = GaussianMixture(n_components=len(np.unique(y_train)), random_state=42)
    gmm_model.fit(X_train)
    gmm_preds = gmm_model.predict(X_test)
    results['GMM'] = accuracy_score(y_test, gmm_preds)
    joblib.dump(gmm_model, 'models/gmm_model.pkl')
    print(f"GMM Accuracy: {results['GMM']:.2f}")

    # Save results
    with open('results/accuracy_results.txt', 'w') as file:
        for model, acc in results.items():
            file.write(f"{model} Accuracy: {acc:.2f}\n")

# Run training and evaluation
if __name__ == "__main__":
    train_and_evaluate_models()
