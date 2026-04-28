import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import yaml
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
    
    epochs = config.get("layer2_ids", {}).get("epochs", 30)
    
    print(f"Training Attention IDS on {device} for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            
            optimizer.zero_grad()
            out = model(batch_x)
            
            loss = criterion(out.squeeze(-1), batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                out = model(batch_x)
                preds = torch.sigmoid(out.squeeze(-1)) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_f1 = calculate_macro_f1(all_targets, all_preds)
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_loader):.4f} | Val Macro F1: {val_f1:.4f}")
        
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/attention_ids.pth")
    print("Attention IDS training complete. Model saved to checkpoints/attention_ids.pth")
    return model

if __name__ == "__main__":
    train_ids()
