import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.79),
    transforms.RandomPosterize(bits=8, p=0.51),
    transforms.ColorJitter(brightness=1.05, contrast=1.07, saturation=0.98, hue=0.0),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
