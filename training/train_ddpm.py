import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import yaml
from models.tab_ddpm import TabDDPM
from data.dataset_loaders import load_dataset

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
    
    epochs = config.get("layer1_ddpm", {}).get("epochs", 50)
    num_timesteps = config.get("layer1_ddpm", {}).get("num_timesteps", 1000)
    
    print(f"Training TabDDPM on {device} for {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x = batch_x.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, num_timesteps, (batch_x.shape[0],), device=device).long()
            
            optimizer.zero_grad()
            pred_noise, true_noise = model(batch_x, t)
            
            loss = criterion(pred_noise, true_noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f}")
        
    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/tab_ddpm.pth")
    print("TabDDPM training complete. Model saved to checkpoints/tab_ddpm.pth")
    return model

if __name__ == "__main__":
    train_ddpm()
