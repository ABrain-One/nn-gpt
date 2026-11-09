import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=1.12, saturation=0.87, hue=0.06),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.31, 1.91)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
