import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=0.99, saturation=0.89, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.12, 1.94)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
