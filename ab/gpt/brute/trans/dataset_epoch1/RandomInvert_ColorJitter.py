import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.52),
    transforms.ColorJitter(brightness=1.02, contrast=0.86, saturation=0.88, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
