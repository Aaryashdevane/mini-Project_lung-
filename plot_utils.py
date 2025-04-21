# File: plot_utils.py

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np


def plot_precision_recall_curves(y_test, y_pred_prob, class_names):
    """
    Plot precision-recall curves for multi-class predictions.
    
    Parameters:
        y_test: array-like, shape (n_samples,) or (n_samples, n_classes)
        y_pred_prob: array-like, shape (n_samples, n_classes)
        class_names: list of class names, length n_classes
    """
    # Determine number of classes
    n_classes = y_pred_prob.shape[1]

    # Binarize y_test if it's a 1D array of class labels
    if y_test.ndim == 1:
        y_test_binarized = label_binarize(y_test, classes=np.arange(n_classes))
    else:
        y_test_binarized = y_test

    # Plot each class
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_test_binarized[:, i], y_pred_prob[:, i]
        )
        plt.plot(recall, precision, lw=2, label=f'{label}')

    # Formatting
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves for Each Class', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
