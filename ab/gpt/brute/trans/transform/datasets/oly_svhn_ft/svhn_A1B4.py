import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=27), transforms.ColorJitter(brightness=1.12, contrast=0.98, saturation=0.85, hue=0.01), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])