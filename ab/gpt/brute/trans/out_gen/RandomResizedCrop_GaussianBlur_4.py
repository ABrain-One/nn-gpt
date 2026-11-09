import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.96), ratio=(0.75, 2.74)),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.83, 1.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
