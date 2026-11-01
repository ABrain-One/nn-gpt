import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.03, contrast=1.1, saturation=1.06, hue=0.07),
    transforms.RandomVerticalFlip(p=0.7),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
