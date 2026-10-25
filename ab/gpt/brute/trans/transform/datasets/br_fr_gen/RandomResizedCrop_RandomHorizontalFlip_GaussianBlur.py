import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.91), ratio=(1.05, 1.36)),
    transforms.RandomHorizontalFlip(p=0.85),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.51, 1.15)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
