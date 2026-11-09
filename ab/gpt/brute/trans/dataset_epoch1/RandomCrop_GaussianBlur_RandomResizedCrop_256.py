import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.42, 1.11)),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.85), ratio=(0.8, 2.55)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
