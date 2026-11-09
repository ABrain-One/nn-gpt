import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.91, contrast=0.87, saturation=1.15, hue=0.0),
    transforms.RandomVerticalFlip(p=0.86),
    transforms.CenterCrop(size=27),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
