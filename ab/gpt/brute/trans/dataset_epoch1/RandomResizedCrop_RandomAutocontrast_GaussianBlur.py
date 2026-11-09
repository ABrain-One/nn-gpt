import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.52, 1.0), ratio=(1.2, 1.79)),
    transforms.RandomAutocontrast(p=0.87),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.83, 1.11)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
