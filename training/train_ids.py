import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import yaml
import mlflow
from models.attention_ids import AttentionIDS
from data.dataset_loaders import load_dataset
from evaluation.metrics import calculate_macro_f1

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

def train_ids(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, test_loader, input_dim, _ = load_dataset(config)
    
    model = AttentionIDS(input_dim, config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("layer2_ids", {}).get("learning_rate", 1e-4)))
    
    alpha = config.get("layer2_ids", {}).get("focal_loss_alpha", 0.25)
    gamma = config.get("layer2_ids", {}).get("focal_loss_gamma", 2.0)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    
    epochs = config.get("layer2_ids", {}).get("epochs", 50)
    use_fp16 = config.get("layer2_ids", {}).get("fp16", True) and torch.cuda.is_available()
    grad_clip = config.get("layer2_ids", {}).get("gradient_clip", 1.0)
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Training Attention IDS on {device} for {epochs} epochs | AMP: {use_fp16} | GradClip: {grad_clip}")
    
    history = {"train_loss": [], "val_f1": []}
    
    try:
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="IDS_Training", nested=True)
        mlflow.log_params({"ids_epochs": epochs, "focal_alpha": alpha, "focal_gamma": gamma, "fp16": use_fp16})
    except:
        pass
        
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                out = model(batch_x)
                loss = criterion(out.squeeze(-1), batch_y)
                
            scaler.scale(loss).backward()
            
            # Gradient clipping for attention stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                # Autocast for faster inference
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    out = model(batch_x)
                preds = torch.sigmoid(out.squeeze(-1)) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_f1 = calculate_macro_f1(all_targets, all_preds)
        history["train_loss"].append(avg_train_loss)
        history["val_f1"].append(val_f1)
        
        print(f"Epoch {epoch+1} Loss: {avg_train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        try:
            mlflow.log_metric("ids_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("ids_val_f1", val_f1, step=epoch)
        except:
            pass
            
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/attention_ids.pth")
    print("Attention IDS training complete. Model saved to checkpoints/attention_ids.pth")
    return model, history

if __name__ == "__main__":
    train_ids()
