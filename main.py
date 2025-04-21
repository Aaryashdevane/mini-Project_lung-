# File: main.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import load_data
from model_utils import train_svm, train_rf, evaluate_model, save_model
from plot_utils import plot_precision_recall_curves


def main():
    # ⚠️ Ensure absolute paths are correct
    audio_dir = (
        '/home/aaryash/.cache/kagglehub/datasets/vbookshelf/'
        'respiratory-sound-database/versions/2/'
        'Respiratory_Sound_Database/Respiratory_Sound_Database/'
        'audio_and_txt_files'
    )
    metadata_file = (
        '/home/aaryash/.cache/kagglehub/datasets/vbookshelf/'
        'respiratory-sound-database/versions/2/'
        'Respiratory_Sound_Database/Respiratory_Sound_Database/'
        'patient_diagnosis.csv'
    )

    # Inspect raw metadata categories
    df_meta = pd.read_csv(metadata_file, header=None, names=['Patient_ID', 'Disease'])
    print("Raw Disease Categories:", df_meta['Disease'].unique())
    print("\nCounts per Category:\n", df_meta['Disease'].value_counts())

    # Load data
    print(f"\nLoading data from:\n  audio_dir = {audio_dir}\n  metadata = {metadata_file}")
    X, y, label_encoder = load_data(audio_dir, metadata_file)

    # Show label encoding mapping
    mapping_df = pd.DataFrame({
        'Encoded_Label': np.arange(len(label_encoder.classes_)),
        'Disease': label_encoder.classes_
    })
    print("\nLabel Encoding Mapping:\n", mapping_df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Train SVM
    print("\nTraining SVM...")
    svm_model = train_svm(X_train, y_train)
    print("Evaluating SVM:")
    evaluate_model(svm_model, X_test, y_test)
    save_model(svm_model, 'svm_model.pkl')

    # Prepare class names and probabilities for SVM
    svm_classes = svm_model.classes_
    svm_class_names = label_encoder.inverse_transform(svm_classes)
    y_pred_prob_svm = svm_model.predict_proba(X_test)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = train_rf(X_train, y_train)
    print("Evaluating Random Forest:")
    evaluate_model(rf_model, X_test, y_test)
    save_model(rf_model, 'rf_model.pkl')

    # Prepare class names and probabilities for RF
    rf_classes = rf_model.classes_
    rf_class_names = label_encoder.inverse_transform(rf_classes)
    y_pred_prob_rf = rf_model.predict_proba(X_test)

    # Plot Precision-Recall Curves
    print("\nPlotting Precision-Recall Curves...")
    plot_precision_recall_curves(y_test, y_pred_prob_svm, svm_class_names)
    plot_precision_recall_curves(y_test, y_pred_prob_rf, rf_class_names)


if __name__ == '__main__':
    main()
