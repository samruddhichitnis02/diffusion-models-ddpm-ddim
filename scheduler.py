import os
import torch
import torch.nn.functional as F

class DiffusionScheduler:
    """
    Manages the forward noising process and variance schedules:
    - Linear noise schedule beta_t: from beta_start to beta_end over num_timesteps
    - Closed-form forward sample q(x_t | x_0)
    - Posterior variance for reverse sampling
    """
    def __init__(self, num_timesteps=100, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precomputed constants for closed-form forward sampling q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior variance for DDPM reverse step: q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
    def q_sample(self, x_0, t, noise=None):
        """
        Closed-form forward diffusion:
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise, noise
