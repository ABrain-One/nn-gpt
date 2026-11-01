import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.GaussianBlur(kernel_size=5, sigma=(0.74, 1.76)),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.81), ratio=(1.1, 2.14)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
