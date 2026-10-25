import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.95, contrast=0.9, saturation=1.19, hue=0.05),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.97, 1.83)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
