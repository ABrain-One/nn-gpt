import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.11),
    transforms.ColorJitter(brightness=0.8, contrast=0.94, saturation=0.85, hue=0.08),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
