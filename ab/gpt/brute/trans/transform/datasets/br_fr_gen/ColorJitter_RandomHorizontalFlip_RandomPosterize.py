import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=0.91, saturation=0.87, hue=0.01),
    transforms.RandomHorizontalFlip(p=0.11),
    transforms.RandomPosterize(bits=8, p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
