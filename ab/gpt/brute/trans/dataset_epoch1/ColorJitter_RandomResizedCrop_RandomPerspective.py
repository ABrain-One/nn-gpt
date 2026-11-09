import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.04, contrast=0.89, saturation=0.85, hue=0.01),
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.91), ratio=(1.21, 2.88)),
    transforms.RandomPerspective(distortion_scale=0.29, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
