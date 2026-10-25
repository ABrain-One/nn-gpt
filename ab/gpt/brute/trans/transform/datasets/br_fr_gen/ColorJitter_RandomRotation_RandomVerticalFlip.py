import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.09, contrast=1.03, saturation=1.12, hue=0.09),
    transforms.RandomRotation(degrees=19),
    transforms.RandomVerticalFlip(p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
