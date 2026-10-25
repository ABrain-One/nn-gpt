import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.82), ratio=(1.28, 2.11)),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.8, 1.66)),
    transforms.RandomAutocontrast(p=0.5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
