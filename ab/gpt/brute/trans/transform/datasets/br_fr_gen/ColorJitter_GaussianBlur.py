import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.89, contrast=0.9, saturation=0.96, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.87, 1.34)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
