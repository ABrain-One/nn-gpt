import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.29),
    transforms.ColorJitter(brightness=0.85, contrast=0.93, saturation=0.91, hue=0.06),
    transforms.RandomAdjustSharpness(sharpness_factor=1.92, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
