import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.13),
    transforms.CenterCrop(size=30),
    transforms.ColorJitter(brightness=0.8, contrast=0.82, saturation=1.16, hue=0.07),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
