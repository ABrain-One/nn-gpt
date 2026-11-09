import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomPosterize(threshold=7, p=0.6),
    transforms.RandomAutocontrast(p=0.79),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])