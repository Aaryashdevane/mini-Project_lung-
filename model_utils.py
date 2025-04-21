# File: model_utils.py

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle


def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine classifier with probability estimates enabled.
    """
    svm_classifier = SVC(kernel='linear', probability=True, random_state=42)
    svm_classifier.fit(X_train, y_train)
    return svm_classifier


def train_rf(X_train, y_train):
    """
    Train a Random Forest classifier.
    """
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the provided model and print accuracy, classification report, and confusion matrix.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{confusion}")
    
    return accuracy, report, confusion


def save_model(model, model_filename):
    """
    Save the trained model to disk using pickle.
    """
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_filename}")


def load_model(model_filename):
    """
    Load a pickled model from disk.
    """
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model
