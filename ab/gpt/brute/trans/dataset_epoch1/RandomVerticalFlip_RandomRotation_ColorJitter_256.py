import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.41),
    transforms.RandomRotation(degrees=22),
    transforms.ColorJitter(brightness=1.1, contrast=0.93, saturation=0.84, hue=0.08),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
