import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.8, 1.97)),
    transforms.ColorJitter(brightness=0.84, contrast=1.13, saturation=0.99, hue=0.05),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
