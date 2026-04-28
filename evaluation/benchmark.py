import torch
import yaml
import numpy as np
from tqdm import tqdm
import json
import os
import mlflow
from models.attention_ids import AttentionIDS, fgsm_attack, purify_data
from models.tab_ddpm import TabDDPM
from data.dataset_loaders import load_dataset
from evaluation.metrics import calculate_macro_f1, calculate_roc_auc, get_detailed_report, plot_confusion_matrix

def run_ablation_studies(config_path="configs/default.yaml"):
    """
    Runs the ablation study framework over multiple seeds.
    Ablations:
    1. Clean inference vs FGSM vs Purified
    Scales to multiple seeds and tracks mean/std robustly.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seeds = config.get("evaluation", {}).get("num_seeds", 5)
    
    _, test_loader, input_dim, _ = load_dataset(config)
    
    # Load models
    ids_model = AttentionIDS(input_dim, config).to(device)
    ddpm_model = TabDDPM(input_dim, config).to(device)
    
    if os.path.exists("checkpoints/attention_ids.pth"):
        ids_model.load_state_dict(torch.load("checkpoints/attention_ids.pth"))
    if os.path.exists("checkpoints/tab_ddpm.pth"):
        ddpm_model.load_state_dict(torch.load("checkpoints/tab_ddpm.pth"))
        
    ids_model.eval()
    
    epsilon = config.get("layer2_ids", {}).get("fgsm_epsilon", 0.1)
    t_purify = config.get("layer2_ids", {}).get("purification_steps", 100)
    use_fp16 = config.get("layer2_ids", {}).get("fp16", True) and torch.cuda.is_available()
    
    results = {"clean_f1": [], "fgsm_f1": [], "purified_f1": [], "clean_auc": [], "purified_auc": []}
    
    print(f"Running Ablation Studies over {num_seeds} seeds...")
    
    # For final reporting, keep the predictions of the last seed
    final_targets, final_clean_preds, final_purified_preds = [], [], []
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        clean_preds, fgsm_preds, purified_preds, targets = [], [], [], []
        clean_probs, purified_probs = [], []
        
        for batch_x, batch_y in tqdm(test_loader, desc=f"Seed {seed+1}/{num_seeds}", leave=False):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            
            # 1. Clean
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out_clean = ids_model(batch_x)
                    probs_c = torch.sigmoid(out_clean.squeeze(-1))
                    preds_c = probs_c > 0.5
                
            # 2. FGSM Attack
            x_adv = fgsm_attack(ids_model, batch_x, batch_y, epsilon)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out_adv = ids_model(x_adv)
                    preds_a = torch.sigmoid(out_adv.squeeze(-1)) > 0.5
                
            # 3. Purified
            x_purified = purify_data(ddpm_model, x_adv, t_purify)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out_pur = ids_model(x_purified)
                    probs_p = torch.sigmoid(out_pur.squeeze(-1))
                    preds_p = probs_p > 0.5
                
            clean_preds.extend(preds_c.cpu().numpy())
            clean_probs.extend(probs_c.cpu().numpy())
            fgsm_preds.extend(preds_a.cpu().numpy())
            purified_preds.extend(preds_p.cpu().numpy())
            purified_probs.extend(probs_p.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
            
        results["clean_f1"].append(calculate_macro_f1(targets, clean_preds))
        results["fgsm_f1"].append(calculate_macro_f1(targets, fgsm_preds))
        results["purified_f1"].append(calculate_macro_f1(targets, purified_preds))
        results["clean_auc"].append(calculate_roc_auc(targets, clean_probs))
        results["purified_auc"].append(calculate_roc_auc(targets, purified_probs))
        
        if seed == num_seeds - 1:
            final_targets = targets
            final_clean_preds = clean_preds
            final_purified_preds = purified_preds

    # Generate and save visual plots for the last seed
    plot_confusion_matrix(final_targets, final_clean_preds, title="Clean Data Confusion Matrix", filename="cm_clean.png")
    plot_confusion_matrix(final_targets, final_purified_preds, title="Purified Data Confusion Matrix", filename="cm_purified.png")
    
    print("\n" + "="*50)
    print("FINAL DETAILED REPORT (Clean Data - Last Seed)")
    print(get_detailed_report(final_targets, final_clean_preds))
    print("="*50)

    # Calculate Mean and Std
    print("\n--- Ablation Results (Mean ± Std) ---")
    summary = {}
    for key in results:
        mean_val = np.mean(results[key])
        std_val = np.std(results[key])
        summary[key] = f"{mean_val:.4f} ± {std_val:.4f}"
        print(f"{key}: {summary[key]}")
        
    with open("results_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    return summary
        
if __name__ == "__main__":
    run_ablation_studies()
