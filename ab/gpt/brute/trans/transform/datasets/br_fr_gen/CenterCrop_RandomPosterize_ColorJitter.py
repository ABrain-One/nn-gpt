import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomPosterize(bits=7, p=0.45),
    transforms.ColorJitter(brightness=1.0, contrast=1.14, saturation=0.86, hue=0.09),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
