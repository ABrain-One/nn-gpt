import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=8, p=0.43),
    transforms.ColorJitter(brightness=1.01, contrast=1.01, saturation=0.85, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
