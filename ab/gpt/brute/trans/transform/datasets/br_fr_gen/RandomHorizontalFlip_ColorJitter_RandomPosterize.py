import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.ColorJitter(brightness=1.19, contrast=0.84, saturation=1.17, hue=0.08),
    transforms.RandomPosterize(bits=6, p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
