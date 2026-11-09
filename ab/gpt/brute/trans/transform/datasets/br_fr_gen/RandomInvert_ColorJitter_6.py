import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.26),
    transforms.ColorJitter(brightness=0.86, contrast=1.16, saturation=0.94, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
