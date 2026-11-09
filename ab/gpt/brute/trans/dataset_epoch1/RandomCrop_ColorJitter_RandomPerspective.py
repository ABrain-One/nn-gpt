import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.ColorJitter(brightness=0.82, contrast=1.14, saturation=0.82, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
