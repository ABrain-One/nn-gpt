import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.92, contrast=1.01, saturation=0.85, hue=0.07),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.97), ratio=(0.82, 1.92)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
