import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.79),
    transforms.ColorJitter(brightness=1.11, contrast=1.03, saturation=1.15, hue=0.07),
    transforms.RandomPosterize(bits=8, p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
