import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.91, contrast=1.05, saturation=0.88, hue=0.06),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.RandomPosterize(bits=6, p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
