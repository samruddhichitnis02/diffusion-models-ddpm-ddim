import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import json
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Modular imports
from scheduler import DiffusionScheduler
from dataset import get_mnist_dataloader
from model import MiniUNet
from sampler import sample_ddpm, sample_ddim

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_forward_diffusion(scheduler, dataloader, output_path="forward_diffusion_vis.png"):
    print("Generating Forward Diffusion Visualization...")
    images, _ = next(iter(dataloader))
    img = images[0:1].to(scheduler.device)
    
    target_timesteps = [0, 10, 30, 60, 99]
    torch.manual_seed(42)
    noise = torch.randn_like(img)
    noisy_images, labels = [], []
    
    for t_val in target_timesteps:
        if t_val == 0:
            x_t = img
            disp_label = "t = 0 (Original)"
        else:
            t_tensor = torch.tensor([t_val], device=scheduler.device)
            x_t, _ = scheduler.q_sample(img, t_tensor, noise=noise)
            disp_label = f"t = {t_val+1 if t_val==99 else t_val}"
            
        x_t_disp = (x_t.squeeze().cpu().clamp(-1, 1) + 1.0) / 2.0
        noisy_images.append(x_t_disp.numpy())
        labels.append(disp_label)
        
    fig, axes = plt.subplots(1, len(target_timesteps), figsize=(15, 3.5))
    for idx, ax in enumerate(axes):
        ax.imshow(noisy_images[idx], cmap='gray')
        ax.set_title(labels[idx], fontsize=13, fontweight='bold', pad=8)
        ax.axis('off')
        
    plt.suptitle("Task (a): Forward Diffusion Process ($x_t = \\sqrt{\\bar{\\alpha}_t}x_0 + \\sqrt{1 - \\bar{\\alpha}_t}\\epsilon$)", 
                 fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved forward diffusion visualization to {output_path}")

def train_mini_ddpm(model, scheduler, dataloader, epochs=50, lr=2e-4, checkpoint_path="mini_ddpm_mnist.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    print(f"Starting training for {epochs} epochs...")
    total_start = time.time()
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        total_loss = 0.0
        for images, _ in dataloader:
            images = images.to(scheduler.device)
            B = images.size(0)
            t = torch.randint(0, scheduler.num_timesteps, (B,), device=scheduler.device).long()
            noise = torch.randn_like(images)
            x_t, _ = scheduler.q_sample(images, t, noise=noise)
            
            pred = model(x_t, t)
            loss = F.mse_loss(pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        ep_time = time.time() - ep_start
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.6f} - Time: {ep_time:.2f}s")
            
    print(f"Training completed in {time.time() - total_start:.2f}s")
    
    # Save checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")
    
    # Save CSV
    with open('training_loss.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'mse_loss'])
        for ep, l in enumerate(loss_history, start=1):
            writer.writerow([ep, l])
    print("Saved training loss to training_loss.csv")
    
    # Plot loss
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, epochs + 1), loss_history, marker='o', markersize=4, color='#1f77b4', linewidth=2, label='Training MSE Loss')
    plt.title("Task (b): Mini DDPM Training Loss Curve (50 Epochs)", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss $L = \\mathbb{E}[||\\epsilon - \\epsilon_\\theta(x_t, t)||^2]$", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()
    return loss_history

def run_benchmarks(model, scheduler):
    print("Running Comparative Sampling & Benchmarks...")
    set_seed(123)
    shape = (16, 1, 28, 28)
    initial_noise = torch.randn(shape, device=scheduler.device)
    
    # DDPM
    ddpm_samples, ddpm_time, _ = sample_ddpm(model, scheduler, shape, initial_noise=initial_noise)
    # DDIM 50
    ddim50_samples, ddim50_time, _ = sample_ddim(model, scheduler, shape, num_steps=50, initial_noise=initial_noise)
    # DDIM 20
    ddim20_samples, ddim20_time, _ = sample_ddim(model, scheduler, shape, num_steps=20, initial_noise=initial_noise)
    
    benchmarks = [
        {'method': 'DDPM', 'steps': 100, 'batch_size': 16, 'total_time_seconds': round(ddpm_time, 3), 'time_per_step_ms': round(ddpm_time/100*1000, 2), 'speedup': 1.0, 'deterministic': False},
        {'method': 'DDIM', 'steps': 50, 'batch_size': 16, 'total_time_seconds': round(ddim50_time, 3), 'time_per_step_ms': round(ddim50_time/50*1000, 2), 'speedup': round(ddpm_time/ddim50_time, 2), 'deterministic': True},
        {'method': 'DDIM', 'steps': 20, 'batch_size': 16, 'total_time_seconds': round(ddim20_time, 3), 'time_per_step_ms': round(ddim20_time/20*1000, 2), 'speedup': round(ddpm_time/ddim20_time, 2), 'deterministic': True}
    ]
    
    with open('sampling_benchmarks.json', 'w') as f:
        json.dump({'benchmarks': benchmarks}, f, indent=4)
    with open('sampling_benchmarks.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'steps', 'batch_size', 'total_time_seconds', 'time_per_step_ms', 'speedup', 'deterministic'])
        writer.writeheader()
        writer.writerows(benchmarks)
    print("Saved sampling benchmarks to sampling_benchmarks.json and sampling_benchmarks.csv")

def main():
    set_seed(42)
    dataloader = get_mnist_dataloader(data_dir='./data', batch_size=128)
    scheduler = DiffusionScheduler(num_timesteps=100, beta_start=0.0001, beta_end=0.02, device=device)
    visualize_forward_diffusion(scheduler, dataloader)
    
    model = MiniUNet(in_channels=1, base_channels=32, time_emb_dim=128).to(device)
    ckpt_path = "mini_ddpm_mnist.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint '{ckpt_path}'...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        train_mini_ddpm(model, scheduler, dataloader, epochs=50, lr=2e-4, checkpoint_path=ckpt_path)
        
    run_benchmarks(model, scheduler)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
