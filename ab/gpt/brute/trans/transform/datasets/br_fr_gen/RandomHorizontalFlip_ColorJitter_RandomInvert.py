import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.61),
    transforms.ColorJitter(brightness=1.03, contrast=0.8, saturation=1.06, hue=0.05),
    transforms.RandomInvert(p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
