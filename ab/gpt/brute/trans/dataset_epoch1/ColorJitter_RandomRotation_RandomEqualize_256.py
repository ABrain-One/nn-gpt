import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.04, contrast=0.93, saturation=1.13, hue=0.04),
    transforms.RandomRotation(degrees=3),
    transforms.RandomEqualize(p=0.57),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
