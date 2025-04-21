# File: data_preprocessing.py
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder

def load_data(audio_dir, metadata_file):
    """
    Load audio data and metadata, extract features, and encode labels.

    Parameters:
    - audio_dir: Path to directory containing .wav files
    - metadata_file: Path to CSV file with patient diagnosis mapping

    Returns:
    - X: numpy array of extracted features (shape: n_samples x n_features)
    - y: numpy array of encoded labels (shape: n_samples,)
    - label_encoder: fitted LabelEncoder instance for inverse transform
    """
    # Load metadata
    df_meta = pd.read_csv(metadata_file, header=None, names=['Patient_ID', 'Disease'])
    label_map = df_meta.set_index('Patient_ID')['Disease'].to_dict()

    features = []
    labels = []

    # Iterate through audio files
    for file_name in os.listdir(audio_dir):
        if file_name.lower().endswith('.wav'):
            try:
                # Extract patient ID from filename
                patient_id = int(file_name.split('_')[0])
                label = label_map.get(patient_id)
                if label is None:
                    continue

                file_path = os.path.join(audio_dir, file_name)
                # Load audio
                y_audio, sr = librosa.load(file_path, sr=None)
                # Feature extraction example: MFCCs (13 coefficients averaged over time)
                mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)

                features.append(mfcc_mean)
                labels.append(label)
            except Exception as e:
                print(f"Skipping {file_name}: {e}")

    # Convert to numpy
    X = np.array(features)
    y_labels = np.array(labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    return X, y, label_encoder
