# scripts/evaluate_ann.py

from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_and_split_data

# Load data
directory_path = 'data'  # Modify according to your data path
X_train, X_test, y_train, y_test = load_and_split_data(directory_path)

# Load the trained ANN model
ann_model = joblib.load('models/ann_model.pkl')

# Make predictions
ann_preds = ann_model.predict(X_test)

# Calculate accuracy
ann_accuracy = accuracy_score(y_test, ann_preds)
print(f"ANN Accuracy on Test Set: {ann_accuracy:.2f}")
