import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.17),
    transforms.ColorJitter(brightness=0.81, contrast=1.17, saturation=0.92, hue=0.04),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
