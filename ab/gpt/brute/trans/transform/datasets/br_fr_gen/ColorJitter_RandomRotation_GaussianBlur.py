import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.99, contrast=1.09, saturation=0.87, hue=0.01),
    transforms.RandomRotation(degrees=16),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.38, 1.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
