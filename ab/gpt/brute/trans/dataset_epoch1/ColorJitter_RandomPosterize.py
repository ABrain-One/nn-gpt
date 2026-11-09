import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=0.87, saturation=1.17, hue=0.02),
    transforms.RandomPosterize(bits=6, p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
