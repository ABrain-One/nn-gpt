import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.94, contrast=1.14, saturation=1.19, hue=0.08),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.85, 1.59)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
