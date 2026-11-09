import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.86), ratio=(0.87, 2.62)),
    transforms.CenterCrop(size=30),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.65, 1.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
