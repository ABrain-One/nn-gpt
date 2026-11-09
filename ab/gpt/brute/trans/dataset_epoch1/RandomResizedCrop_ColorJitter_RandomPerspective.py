import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.95), ratio=(0.88, 2.22)),
    transforms.ColorJitter(brightness=1.02, contrast=1.11, saturation=0.87, hue=0.08),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
