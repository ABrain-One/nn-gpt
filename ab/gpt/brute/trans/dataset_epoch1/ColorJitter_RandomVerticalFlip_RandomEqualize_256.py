import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.82, contrast=1.18, saturation=0.98, hue=0.02),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomEqualize(p=0.7),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
