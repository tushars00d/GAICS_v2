import torch
import yaml
import numpy as np
from tqdm import tqdm
import json
import os
from models.attention_ids import AttentionIDS, fgsm_attack, purify_data
from models.tab_ddpm import TabDDPM
from data.dataset_loaders import load_dataset
from evaluation.metrics import calculate_macro_f1

def run_ablation_studies(config_path="configs/default.yaml"):
    """
    Runs the ablation study framework over multiple seeds.
    Ablations:
    1. Clean inference vs FGSM vs Purified
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_seeds = config.get("evaluation", {}).get("num_seeds", 5)
    
    _, test_loader, input_dim, _ = load_dataset(config)
    
    # Load models
    ids_model = AttentionIDS(input_dim, config).to(device)
    ddpm_model = TabDDPM(input_dim, config).to(device)
    
    # Load checkpoints if they exist
    if os.path.exists("checkpoints/attention_ids.pth"):
        ids_model.load_state_dict(torch.load("checkpoints/attention_ids.pth"))
    if os.path.exists("checkpoints/tab_ddpm.pth"):
        ddpm_model.load_state_dict(torch.load("checkpoints/tab_ddpm.pth"))
        
    ids_model.eval()
    
    epsilon = config.get("layer2_ids", {}).get("fgsm_epsilon", 0.1)
    t_purify = config.get("layer2_ids", {}).get("purification_steps", 100)
    
    results = {"clean_f1": [], "fgsm_f1": [], "purified_f1": []}
    
    print(f"Running Ablation Studies over {num_seeds} seeds...")
    
    for seed in range(num_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        clean_preds, fgsm_preds, purified_preds, targets = [], [], [], []
        
        for batch_x, batch_y in tqdm(test_loader, desc=f"Seed {seed+1}/{num_seeds}", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 1. Clean
            with torch.no_grad():
                out_clean = ids_model(batch_x)
                preds_c = torch.sigmoid(out_clean.squeeze(-1)) > 0.5
                
            # 2. FGSM Attack
            x_adv = fgsm_attack(ids_model, batch_x, batch_y, epsilon)
            with torch.no_grad():
                out_adv = ids_model(x_adv)
                preds_a = torch.sigmoid(out_adv.squeeze(-1)) > 0.5
                
            # 3. Purified
            x_purified = purify_data(ddpm_model, x_adv, t_purify)
            with torch.no_grad():
                out_pur = ids_model(x_purified)
                preds_p = torch.sigmoid(out_pur.squeeze(-1)) > 0.5
                
            clean_preds.extend(preds_c.cpu().numpy())
            fgsm_preds.extend(preds_a.cpu().numpy())
            purified_preds.extend(preds_p.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())
            
        results["clean_f1"].append(calculate_macro_f1(targets, clean_preds))
        results["fgsm_f1"].append(calculate_macro_f1(targets, fgsm_preds))
        results["purified_f1"].append(calculate_macro_f1(targets, purified_preds))

    # Calculate Mean and Std
    print("\n--- Ablation Results ---")
    for key in results:
        mean_val = np.mean(results[key])
        std_val = np.std(results[key])
        print(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
        
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    run_ablation_studies()
