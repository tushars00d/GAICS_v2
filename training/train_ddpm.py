import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import yaml
from models.tab_ddpm import TabDDPM
from data.dataset_loaders import load_dataset
import mlflow

def train_ddpm(config_path="configs/default.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Data
    train_loader, _, input_dim, _ = load_dataset(config)
    
    # Init Model
    model = TabDDPM(input_dim, config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.get("layer1_ddpm", {}).get("learning_rate", 1e-4)))
    criterion = nn.MSELoss()
    
    epochs = config.get("layer1_ddpm", {}).get("epochs", 100)
    num_timesteps = config.get("layer1_ddpm", {}).get("num_timesteps", 1000)
    use_fp16 = config.get("layer1_ddpm", {}).get("fp16", True) and torch.cuda.is_available()
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    
    # Learning Rate Scheduler for stable convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Training TabDDPM on {device} for {epochs} epochs | AMP: {use_fp16}")
    model.train()
    
    history = {"loss": [], "lr": []}
    
    # Optionally start an MLflow run
    try:
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="DDPM_Training", nested=True)
        mlflow.log_params({"ddpm_epochs": epochs, "ddpm_lr": float(config.get("layer1_ddpm", {}).get("learning_rate", 1e-4)), "fp16": use_fp16})
    except:
        pass
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device, non_blocking=True)
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (batch_x.shape[0],), device=device).long()
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=use_fp16):
                pred_noise, true_noise = model(batch_x, t)
                loss = criterion(pred_noise, true_noise)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        history["loss"].append(avg_loss)
        history["lr"].append(current_lr)
        
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
        try:
            mlflow.log_metric("ddpm_loss", avg_loss, step=epoch)
            mlflow.log_metric("ddpm_lr", current_lr, step=epoch)
        except:
            pass
            
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/tab_ddpm.pth")
    print("TabDDPM training complete. Model saved to checkpoints/tab_ddpm.pth")
    return model, history

if __name__ == "__main__":
    train_ddpm()
