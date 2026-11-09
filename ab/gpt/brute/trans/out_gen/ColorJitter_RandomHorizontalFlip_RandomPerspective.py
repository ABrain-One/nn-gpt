import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.12, contrast=0.93, saturation=1.16, hue=0.01),
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
