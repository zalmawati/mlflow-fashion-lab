import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Mean/std of Fashion-MNIST pixels, normalizing helps training converge
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

def get_loaders(batch_size=64, val_size=5000, data_dir="data", seed=42):
    full_train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=TRANSFORM)
    gen = torch.Generator().manual_seed(seed) #same split every time = reproducible
    train, val = random_split(full_train, [len(full_train)-val_size, val_size], generator=gen)
    return (DataLoader(train, batch_size, shuffle=True),
            DataLoader(val, batch_size=batch_size),
            DataLoader(test, batch_size=batch_size))
    