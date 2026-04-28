import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A multi-layer perceptron for the reverse diffusion process.
    Predicts the noise added to the input tabular data at timestep t.
    """
    def __init__(self, input_dim, hidden_dims=[256, 512, 256]):
        super().__init__()
        layers = []
        
        # Input dim + 1 for the timestep embedding
        dim = input_dim + 1 
        for h_dim in hidden_dims:
            layers.append(nn.Linear(dim, h_dim))
            layers.append(nn.SiLU())
            dim = h_dim
            
        layers.append(nn.Linear(dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        # t is shape (batch_size,)
        t = t.unsqueeze(-1).float()
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

class TabDDPM(nn.Module):
    """
    Tabular Denoising Diffusion Probabilistic Model (Layer 1).
    Synthesizes tabular features.
    """
    def __init__(self, input_dim, config):
        super().__init__()
        self.num_timesteps = config.get("layer1_ddpm", {}).get("num_timesteps", 1000)
        self.model = MLP(input_dim, config.get("layer1_ddpm", {}).get("hidden_dims", [256, 512, 256]))
        
        # Beta schedule (linear for simplicity, can be changed to cosine)
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.num_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def forward_diffusion(self, x0, t, noise=None):
        """
        q(x_t | x_0)
        Adds noise to the original sample x0 at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod[t])[:, None]
        
        xt = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise
        return xt, noise

    def forward(self, x0, t, noise=None):
        """
        Forward pass for training: predicting the noise.
        """
        xt, noise = self.forward_diffusion(x0, t, noise)
        pred_noise = self.model(xt, t)
        return pred_noise, noise

    @torch.no_grad()
    def sample(self, num_samples, input_dim, device):
        """
        Reverse diffusion process to generate synthetic data.
        """
        self.model.eval()
        x = torch.randn(num_samples, input_dim).to(device)
        
        for i in reversed(range(self.num_timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.model(x, t)
            
            alpha = self.alphas[t][:, None]
            alpha_hat = self.alphas_cumprod[t][:, None]
            beta = self.betas[t][:, None]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
            
        self.model.train()
        return x
