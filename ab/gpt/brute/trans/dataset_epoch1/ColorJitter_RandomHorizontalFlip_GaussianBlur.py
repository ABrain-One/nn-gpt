import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.13, contrast=0.82, saturation=0.96, hue=0.09),
    transforms.RandomHorizontalFlip(p=0.24),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.44, 1.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
