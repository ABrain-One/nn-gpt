import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.87, contrast=0.93, saturation=0.86, hue=0.09),
    transforms.RandomRotation(degrees=25),
    transforms.RandomInvert(p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
