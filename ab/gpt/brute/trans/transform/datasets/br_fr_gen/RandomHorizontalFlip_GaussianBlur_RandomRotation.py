import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.88),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.95, 1.24)),
    transforms.RandomRotation(degrees=14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
