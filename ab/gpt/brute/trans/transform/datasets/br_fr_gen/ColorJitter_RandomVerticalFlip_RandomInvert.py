import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.17, contrast=1.03, saturation=0.88, hue=0.01),
    transforms.RandomVerticalFlip(p=0.64),
    transforms.RandomInvert(p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
