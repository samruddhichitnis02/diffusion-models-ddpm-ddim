import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_mnist_dataloader(data_dir="./data", batch_size=128, shuffle=True, download=False):
    """
    Loads MNIST dataset normalized to [-1, 1] as standard in DDPM.
    Returns: DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # scale from [0, 1] to [-1, 1]
    ])
    
    dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader
