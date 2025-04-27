import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from joblib import load

from data_preprocessing import load_data
from model_utils import train_svm, train_rf, evaluate_model, save_model, train_knn
from plot_utils import plot_precision_recall_curves, plot_model_comparison

# CNN model definition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ANN model definition
def build_ann_model(input_shape, num_classes=6):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # File paths
    audio_dir = '/home/aaryash/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/versions/2/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
    metadata_file = '/home/aaryash/.cache/kagglehub/datasets/vbookshelf/respiratory-sound-database/versions/2/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv'

    # Load metadata and display basic information
    df_meta = pd.read_csv(metadata_file, header=None, names=['Patient_ID', 'Disease'])
    print("Raw Disease Categories:", df_meta['Disease'].unique())
    print("\nCounts per Category:\n", df_meta['Disease'].value_counts())

    # Load features and labels from the dataset
    X, y, label_encoder = load_data(audio_dir, metadata_file)

    mapping_df = pd.DataFrame({
        'Encoded_Label': np.arange(len(label_encoder.classes_)),
        'Disease': label_encoder.classes_
    })
    print("\nLabel Encoding Mapping:\n", mapping_df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Convert labels to categorical (one-hot encoding)
    y_train_cat = to_categorical(y_train, num_classes=len(label_encoder.classes_))
    y_test_cat = to_categorical(y_test, num_classes=len(label_encoder.classes_))

    results = {}
    pred_probs = {}
    class_names = label_encoder.classes_

    # Reshape X_train and X_test for models expecting 2D input (SVM, RF, KNN)
    X_train_2d = X_train.reshape(X_train.shape[0], -1)
    X_test_2d = X_test.reshape(X_test.shape[0], -1)

    # SVM
    print("\nTraining SVM...")
    svm_model = train_svm(X_train_2d, y_train)
    print("Evaluating SVM:")
    acc, _, _ = evaluate_model(svm_model, X_test_2d, y_test)
    save_model(svm_model, 'models/svm_model.pkl')
    results['SVM'] = acc * 100
    pred_probs['SVM'] = svm_model.predict_proba(X_test_2d)

    # RF
    print("\nTraining Random Forest...")
    rf_model = train_rf(X_train_2d, y_train)
    print("Evaluating RF:")
    acc, _, _ = evaluate_model(rf_model, X_test_2d, y_test)
    save_model(rf_model, 'models/rf_model.pkl')
    results['RF'] = acc * 100
    pred_probs['RF'] = rf_model.predict_proba(X_test_2d)

    # KNN
    print("\nTraining KNN...")
    knn_model = train_knn(X_train_2d, y_train)
    print("Evaluating KNN:")
    acc, _, _ = evaluate_model(knn_model, X_test_2d, y_test)
    save_model(knn_model, 'models/knn_model.pkl')
    results['KNN'] = acc * 100
    pred_probs['KNN'] = knn_model.predict_proba(X_test_2d)

    # ANN (Keras)
    print("\nTraining ANN...")

    ann_model = build_ann_model(X_train_2d.shape[1], num_classes=y_train_cat.shape[1])
    ann_model.fit(X_train_2d, y_train_cat, epochs=30, batch_size=32, verbose=1)
    ann_acc = ann_model.evaluate(X_test_2d, y_test_cat, verbose=0)[1]
    print(f"ANN Accuracy: {ann_acc:.2f}")
    results['ANN'] = ann_acc * 100
    pred_probs['ANN'] = ann_model.predict(X_test_2d)

    # CNN (Keras)
    print("\nTraining CNN...")

    # Reshape for CNN: (samples, height, width, channels)
    X_train_cnn = X_train.reshape(-1, 40, 174, 1)  # Adjust this based on your input shape
    X_test_cnn = X_test.reshape(-1, 40, 174, 1)

    cnn_model = build_cnn_model((40, 174, 1), num_classes=y_train_cat.shape[1])
    cnn_model.fit(X_train_cnn, y_train_cat, epochs=30, batch_size=32, verbose=1)
    cnn_acc = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)[1]
    print(f"CNN Accuracy: {cnn_acc:.2f}")
    results['CNN'] = cnn_acc * 100
    pred_probs['CNN'] = cnn_model.predict(X_test_cnn)

    # Plot PR Curves
    print("\nPlotting PR Curves...")
    for model in ['SVM', 'RF', 'KNN', 'ANN', 'CNN']:
        plot_precision_recall_curves(y_test, pred_probs[model], class_names)

    # Summary
    print("\n=== Model Accuracy Summary ===")
    for model, acc in results.items():
        print(f"{model} Accuracy: {acc:.2f}%")

    print("\n--- Reference Results ---")
    print("CNN (MFCC+LBP) [5]       : 95.56%")
    print("CNN STFT+MFCC (ICBHI) [6]: 81.62%")
    print("SVM (RALE) [7]           : 75.00%")

    plot_model_comparison(results.get('SVM', 0), results.get('RF', 0),results.get('KNN', 0), results.get('ANN', 0), results.get('CNN', 0))


if __name__ == '__main__':
    main()
