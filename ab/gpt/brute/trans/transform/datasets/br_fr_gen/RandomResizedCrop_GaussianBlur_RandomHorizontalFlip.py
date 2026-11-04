import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.85), ratio=(1.06, 2.15)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.41, 1.24)),
    transforms.RandomHorizontalFlip(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
