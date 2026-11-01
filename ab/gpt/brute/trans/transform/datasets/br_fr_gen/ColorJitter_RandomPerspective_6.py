import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.92, contrast=1.05, saturation=0.89, hue=0.07),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.85),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
