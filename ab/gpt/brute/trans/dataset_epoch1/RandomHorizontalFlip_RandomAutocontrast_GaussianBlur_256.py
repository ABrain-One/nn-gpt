import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomAutocontrast(p=0.87),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.85, 1.17)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
