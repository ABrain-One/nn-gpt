import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.ColorJitter(brightness=0.92, contrast=1.07, saturation=0.98, hue=0.1),
    transforms.RandomPosterize(bits=7, p=0.21),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
