import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.35, 1.65)),
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.93), ratio=(0.83, 2.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
