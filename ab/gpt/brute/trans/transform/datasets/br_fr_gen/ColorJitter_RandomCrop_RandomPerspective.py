import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.87, contrast=1.01, saturation=0.93, hue=0.06),
    transforms.RandomCrop(size=26),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
