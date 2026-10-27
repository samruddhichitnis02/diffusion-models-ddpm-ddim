import time
import torch

@torch.no_grad()
def sample_ddpm(model, scheduler, shape, initial_noise=None, record_trajectories=False):
    """
    DDPM Reverse Sampling:
    x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * eps_theta) + sigma_t * z
    where sigma_t^2 = beta_tilde_t
    """
    model.eval()
    device = scheduler.device
    
    if initial_noise is not None:
        x_t = initial_noise.clone().to(device)
    else:
        x_t = torch.randn(shape, device=device)
        
    trajectories = [x_t.clone().cpu()] if record_trajectories else []
    T = scheduler.num_timesteps
    start_time = time.time()
    
    for t_idx in reversed(range(T)):
        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        pred_noise = model(x_t, t_batch)
        
        alpha_t = scheduler.alphas[t_idx]
        alpha_bar_t = scheduler.alphas_cumprod[t_idx]
        
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
        )
        
        if t_idx > 0:
            sigma_t = torch.sqrt(scheduler.posterior_variance[t_idx])
            z = torch.randn_like(x_t)
            x_t = mean + sigma_t * z
        else:
            x_t = mean
            
        if record_trajectories and (t_idx in [T-1, int(0.75*T), int(0.5*T), int(0.25*T), 0]):
            trajectories.append(x_t.clone().cpu())
            
    elapsed_time = time.time() - start_time
    return x_t, elapsed_time, trajectories

@torch.no_grad()
def sample_ddim(model, scheduler, shape, num_steps=50, eta=0.0, initial_noise=None, record_trajectories=False):
    """
    DDIM Accelerated Reverse Sampling (Deterministic ODE trajectory when eta=0.0):
    1. x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
    2. dir_xt = sqrt(1 - alpha_bar_prev - sigma^2) * eps_theta
    3. x_prev = sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * z
    """
    model.eval()
    device = scheduler.device
    
    if initial_noise is not None:
        x_t = initial_noise.clone().to(device)
    else:
        x_t = torch.randn(shape, device=device)
        
    T = scheduler.num_timesteps
    timesteps = torch.linspace(0, T - 1, steps=num_steps).long().tolist()
    timesteps_prev = [-1] + timesteps[:-1]
    
    trajectories = [x_t.clone().cpu()] if record_trajectories else []
    start_time = time.time()
    
    for i in reversed(range(num_steps)):
        t_curr = timesteps[i]
        t_prev = timesteps_prev[i]
        
        t_batch = torch.full((shape[0],), t_curr, device=device, dtype=torch.long)
        pred_noise = model(x_t, t_batch)
        
        alpha_bar_t = scheduler.alphas_cumprod[t_curr]
        alpha_bar_prev = scheduler.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
        
        # 1. Estimate clean x_0
        x0_pred = (x_t - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        # 2. DDIM variance sigma_t
        if t_prev < 0 or eta == 0.0:
            sigma_t = 0.0
        else:
            sigma_t = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            
        # 3. Direction pointing to x_prev
        dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=0.0)) * pred_noise
        noise_term = sigma_t * torch.randn_like(x_t) if sigma_t > 0 else 0.0
        
        # 4. Step to x_prev
        x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + noise_term
        
        if record_trajectories and (i in [num_steps-1, int(0.75*num_steps), int(0.5*num_steps), int(0.25*num_steps), 0]):
            trajectories.append(x_t.clone().cpu())
            
    elapsed_time = time.time() - start_time
    return x_t, elapsed_time, trajectories
