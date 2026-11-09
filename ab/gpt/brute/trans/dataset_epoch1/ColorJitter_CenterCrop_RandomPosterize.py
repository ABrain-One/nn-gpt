import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.02, contrast=0.96, saturation=0.89, hue=0.06),
    transforms.CenterCrop(size=32),
    transforms.RandomPosterize(bits=7, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
