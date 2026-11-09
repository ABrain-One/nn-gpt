import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.24, p=0.42),
    transforms.ColorJitter(brightness=0.82, contrast=1.18, saturation=0.85, hue=0.0),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
