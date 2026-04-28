from sklearn.metrics import f1_score, roc_auc_score
import numpy as np
import shap
import torch

def calculate_macro_f1(y_true, y_pred):
    """MANDATORY metric for imbalanced datasets."""
    return f1_score(y_true, y_pred, average='macro')

def calculate_roc_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)

def calculate_tabular_fid(real_data, synthetic_data):
    """
    Approximation of Frechet Inception Distance for tabular data.
    Uses the Frechet Distance between feature statistics.
    """
    mu_real, sigma_real = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)
    mu_syn, sigma_syn = np.mean(synthetic_data, axis=0), np.cov(synthetic_data, rowvar=False)
    
    diff = mu_real - mu_syn
    # Approximation without the matrix square root for stability on non-PSD matrices
    # A full strict FID would use scipy.linalg.sqrtm
    covmean = np.trace(sigma_real + sigma_syn - 2 * np.sqrt(np.abs(sigma_real.dot(sigma_syn))))
    return diff.dot(diff) + covmean

def generate_shap_explanations(model, data_sample):
    """
    Layer 5: SHAP-based feature attribution.
    Requires a subset of data as background.
    """
    device = data_sample.device
    model.eval()
    
    # We use a DeepExplainer or GradientExplainer.
    # For PyTorch, GradientExplainer is robust.
    explainer = shap.GradientExplainer(model, data_sample)
    shap_values = explainer.shap_values(data_sample)
    return shap_values
