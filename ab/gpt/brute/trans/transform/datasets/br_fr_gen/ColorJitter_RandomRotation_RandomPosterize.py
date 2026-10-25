import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.1, contrast=0.99, saturation=1.05, hue=0.05),
    transforms.RandomRotation(degrees=7),
    transforms.RandomPosterize(bits=4, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
