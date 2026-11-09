import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.92, contrast=1.13, saturation=0.99, hue=0.07),
    transforms.RandomVerticalFlip(p=0.36),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
