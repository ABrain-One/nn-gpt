import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.82), ratio=(0.95, 2.14)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.34, 1.27)),
    transforms.RandomRotation(degrees=8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
