import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomPosterize(bits=8, p=0.18),
    transforms.ColorJitter(brightness=1.11, contrast=0.83, saturation=1.03, hue=0.06),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
