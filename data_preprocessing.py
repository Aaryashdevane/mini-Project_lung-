# Updated data_preprocessing.py
import os
import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc  # shape: (n_mfcc, max_len)


def load_data(audio_dir, metadata_file):
    features = []
    labels = []

    df_meta = pd.read_csv(metadata_file, header=None, names=['Patient_ID', 'Disease'])
    patient_disease_map = dict(zip(df_meta['Patient_ID'].astype(str), df_meta['Disease']))

    print(f"Scanning directory: {audio_dir}")

    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            file_path = os.path.join(audio_dir, file)
            print(f"Loading: {file_path}")
            mfcc_features = extract_mfcc(file_path)
            features.append(mfcc_features)

            patient_id = file.split('_')[0]
            label = patient_disease_map.get(patient_id, 'Unknown')
            labels.append(label)

    print(f"Total samples loaded: {len(features)}")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    return np.array(features), y_encoded, label_encoder

def load_and_split_data(audio_dir, metadata_file, test_size=0.25):
    X, y, label_encoder = load_data(audio_dir, metadata_file)

    if len(X) == 0:
        raise ValueError("No data found. Check your audio directory and metadata file.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat, label_encoder
