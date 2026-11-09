import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.7),
    transforms.RandomEqualize(p=0.7),
    transforms.ColorJitter(brightness=1.17, contrast=1.04, saturation=1.1, hue=0.02),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
