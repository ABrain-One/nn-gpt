import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.ColorJitter(brightness=1.17, contrast=1.18, saturation=1.01, hue=0.09),
    transforms.RandomVerticalFlip(p=0.44),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
