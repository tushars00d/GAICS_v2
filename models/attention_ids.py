import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionIDS(nn.Module):
    """
    Attention-based Intrusion Detection System (Layer 2).
    Uses multi-head self-attention on tabular features.
    """
    def __init__(self, input_dim, config):
        super().__init__()
        hidden_dims = config.get("layer2_ids", {}).get("hidden_dims", [128, 64])
        num_heads = config.get("layer2_ids", {}).get("num_heads", 4)
        
        self.embedding = nn.Linear(input_dim, hidden_dims[0])
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dims[0], num_heads=num_heads, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1) # Binary classification
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Treat features as a sequence of length 1 for attention
        x_emb = self.embedding(x).unsqueeze(1) # (batch_size, 1, hidden_dim)
        attn_out, _ = self.attention(x_emb, x_emb, x_emb) # (batch_size, 1, hidden_dim)
        
        out = self.fc(attn_out.squeeze(1)) # (batch_size, 1)
        return out


def fgsm_attack(model, x, y, epsilon):
    """
    Fast Gradient Sign Method (FGSM) adversarial attack.
    """
    x_adv = x.clone().detach().requires_grad_(True)
    out = model(x_adv)
    
    # Binary cross entropy with logits
    loss = F.binary_cross_entropy_with_logits(out.squeeze(-1), y.float())
    model.zero_grad()
    loss.backward()
    
    data_grad = x_adv.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create perturbed data
    perturbed_x = x_adv + epsilon * sign_data_grad
    return perturbed_x.detach()


def purify_data(ddpm_model, x_adv, t_purify):
    """
    Diffusion-based adversarial purification.
    Adds noise up to timestep t_purify, then denoises.
    """
    device = x_adv.device
    batch_size = x_adv.shape[0]
    
    # 1. Forward diffusion up to t_purify
    t = torch.full((batch_size,), t_purify, device=device, dtype=torch.long)
    x_noisy, _ = ddpm_model.forward_diffusion(x_adv, t)
    
    # 2. Reverse diffusion from t_purify down to 0
    ddpm_model.eval()
    x = x_noisy
    with torch.no_grad():
        for i in reversed(range(t_purify)):
            t_rev = torch.full((batch_size,), i, device=device, dtype=torch.long)
            pred_noise = ddpm_model.model(x, t_rev)
            
            alpha = ddpm_model.alphas[t_rev][:, None]
            alpha_hat = ddpm_model.alphas_cumprod[t_rev][:, None]
            beta = ddpm_model.betas[t_rev][:, None]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
            
    ddpm_model.train()
    return x
