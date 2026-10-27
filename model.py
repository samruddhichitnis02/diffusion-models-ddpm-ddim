import math
import torch
import torch.nn as nn

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for conditioning on continuous/discrete timesteps.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: (B,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings)
        embeddings = time.unsqueeze(1).float() * embeddings.unsqueeze(0)
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings

class ConvBlock(nn.Module):
    """
    Convolutional block containing:
    Conv2d -> GroupNorm -> ReLU -> Time Projection Addition -> Conv2d -> GroupNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, t_emb):
        h = self.relu1(self.norm1(self.conv1(x)))
        time_proj = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_proj
        h = self.relu2(self.norm2(self.conv2(h)))
        return h

class MiniUNet(nn.Module):
    """
    Mini U-Net noise-prediction network:
    - 10 convolutional layers (exceeding the >= 4 requirement)
    - Nonlinear ReLU activations
    - Conditioned on timestep t
    - Preserves input resolution: (B, 1, 28, 28) -> (B, 1, 28, 28)
    - Skip connections between encoder and decoder stages
    """
    def __init__(self, in_channels=1, base_channels=32, time_emb_dim=128):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder (Downsampling)
        self.down1 = ConvBlock(in_channels, base_channels, time_emb_dim)       # 28x28
        self.pool1 = nn.MaxPool2d(2)                                           # 14x14
        self.down2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim) # 14x14
        self.pool2 = nn.MaxPool2d(2)                                           # 7x7
        
        # Bottleneck
        self.bot = ConvBlock(base_channels * 2, base_channels * 4, time_emb_dim) # 7x7
        
        # Decoder (Upsampling)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 14x14
        self.conv_up1 = ConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 28x28
        self.conv_up2 = ConvBlock(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        
        # Final output conv
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        # Encoder
        d1 = self.down1(x, t_emb)
        p1 = self.pool1(d1)
        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)
        
        # Bottleneck
        b = self.bot(p2, t_emb)
        
        # Decoder with skip connections
        u1 = self.up1(b)
        c1 = torch.cat([u1, d2], dim=1)
        dec1 = self.conv_up1(c1, t_emb)
        
        u2 = self.up2(dec1)
        c2 = torch.cat([u2, d1], dim=1)
        dec2 = self.conv_up2(c2, t_emb)
        
        return self.out_conv(dec2)
