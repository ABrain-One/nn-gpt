import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.93, contrast=1.0, saturation=0.94, hue=0.08),
    transforms.RandomCrop(size=27),
    transforms.RandomInvert(p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
