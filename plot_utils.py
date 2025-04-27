# ================= plot_utils.py =================
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_precision_recall_curves(y_test, y_pred_prob, class_names):
    """
    Plots Precision-Recall curves for each class present in y_test.
    """
    # determine which classes actually appear in y_test
    present_classes = np.unique(y_test)
    n_classes = len(present_classes)

    # binarize only over the present classes
    y_test_binarized = label_binarize(y_test, classes=present_classes)

    # check that y_pred_prob matches the number of present classes
    if y_pred_prob.shape[1] != n_classes:
        print(f"[Warning] Skipping PR curve: y_pred_prob has {y_pred_prob.shape[1]} columns, "
              f"but only {n_classes} classes are present in y_test.")
        return

    plt.figure(figsize=(12, 8))
    for idx, cls in enumerate(present_classes):
        precision, recall, _ = precision_recall_curve(
            y_test_binarized[:, idx],
            y_pred_prob[:, idx]
        )
        plt.plot(recall, precision, lw=2, label=f"{class_names[cls]}")

    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curves for Present Classes', fontsize=16)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()


def plot_model_comparison(model_names, accuracies):
    """
    Plots a bar chart comparing accuracies of multiple models.
    - model_names: list of strings
    - accuracies: list of floats (same length as model_names)
    """
    assert len(model_names) == len(accuracies), "names & accuracies must match length"

    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies,
                   color=['blue','green','magenta','purple','cyan','orange','red','gray'][:len(model_names)])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 100)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.2f}%", ha='center')

    plt.tight_layout()
    plt.show()
