import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomAdjustSharpness(sharpness_factor=1.24, p=0.2),
    transforms.ColorJitter(brightness=0.91, contrast=1.01, saturation=1.02, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
