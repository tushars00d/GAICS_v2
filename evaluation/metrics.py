from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import numpy as np
import shap
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_macro_f1(y_true, y_pred):
    """MANDATORY metric for imbalanced datasets."""
    return f1_score(y_true, y_pred, average='macro')

def calculate_roc_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)

def calculate_tabular_fid(real_data, synthetic_data):
    """
    Approximation of Frechet Inception Distance for tabular data.
    """
    mu_real, sigma_real = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu_syn, sigma_syn = np.mean(synthetic_data, axis=0), np.cov(synthetic_data, rowvar=False)
    
    diff = mu_real - mu_syn
    covmean = np.trace(sigma_real + sigma_syn - 2 * np.sqrt(np.abs(sigma_real.dot(sigma_syn))))
    return diff.dot(diff) + covmean

def generate_shap_explanations(model, data_sample):
    """
    Layer 5: SHAP-based feature attribution.
    """
    device = data_sample.device
    model.eval()
    explainer = shap.GradientExplainer(model, data_sample)
    shap_values = explainer.shap_values(data_sample)
    return shap_values

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()

def plot_roc_curve(y_true, y_prob, title="ROC Curve", filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = calculate_roc_auc(y_true, y_prob)
    
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}")
    plt.close()

def get_detailed_report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=["Normal", "Attack"])
