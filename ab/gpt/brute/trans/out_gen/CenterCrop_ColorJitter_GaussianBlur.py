import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.ColorJitter(brightness=0.87, contrast=0.9, saturation=0.84, hue=0.06),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.71, 1.22)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
