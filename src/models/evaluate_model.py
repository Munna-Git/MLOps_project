#src/models/evaluate_model.py

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    # 🔑 Added imports for ROC and PR curves
    roc_curve,
    auc,
    precision_recall_curve,
)
import seaborn as sns
import matplotlib.pyplot as plt



def calculate_f2_score(precision: float, recall: float) -> float:
    """
    Manually calculate the F2-Score using precision and recall.
    
    Parameters:
    - precision (float): Precision of the model.
    - recall (float): Recall of the model.
    
    Returns:
    - float: F2-Score.
    """
    beta = 2
    if precision + recall == 0:  # Avoid division by zero
        return 0.0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


def get_model_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate and return model performance metrics.
    
    Parameters:
    - y_true (np.ndarray): True labels of the validation dataset.
    - y_pred (np.ndarray): Predicted labels from the model.

    Returns:
    - dict: A dictionary containing evaluation metrics (F1-Score, Precision, Recall, F2-Score, Classification Report).
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f2_score = calculate_f2_score(precision, recall)

    metrics = {
        'f1_score': f1_score(y_true, y_pred),
        'f2_score': f2_score,
        'precision': precision,
        'recall': recall,
        'classification_report': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }

    return metrics


def plot_model_evaluation(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, model_name: str) -> None:
    """
    Generate and display model evaluation visualizations.
    
    Parameters:
    - y_true (np.ndarray): True labels of the validation dataset.
    - y_pred (np.ndarray): Predicted labels from the model.
    - y_prob (np.ndarray): Predicted probabilities of the positive class.
    - model_name (str): Name of the model being evaluated.

    Returns:
    - None: Displays the confusion matrix, ROC curve, and Precision-Recall curve.
    """
    # 1. Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 2. ROC Curve and AUC
    # Calculate ROC curve components (False Positive Rate, True Positive Rate)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 3, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title(f'Receiver Operating Characteristic (ROC) for {model_name}')
    plt.legend(loc="lower right")

    # 3. Precision-Recall Curve
    # Calculate Precision-Recall curve components
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    
    # Calculate the area under the PR curve (AUPRC)
    pr_auc = auc(recall_curve, precision_curve)

    plt.subplot(1, 3, 3)
    plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.show()
    