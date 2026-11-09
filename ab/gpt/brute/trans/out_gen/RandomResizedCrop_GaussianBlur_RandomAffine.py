import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.97), ratio=(1.27, 1.45)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.34, 1.12)),
    transforms.RandomAffine(degrees=3, translate=(0.11, 0.18), scale=(0.81, 1.41), shear=(0.2, 7.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
