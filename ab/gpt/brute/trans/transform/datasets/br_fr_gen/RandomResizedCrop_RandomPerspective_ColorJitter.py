import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.95), ratio=(1.0, 2.67)),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.81),
    transforms.ColorJitter(brightness=1.02, contrast=1.01, saturation=0.89, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
