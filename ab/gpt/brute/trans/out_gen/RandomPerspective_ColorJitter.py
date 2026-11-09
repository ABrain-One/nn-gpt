import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.2, p=0.74),
    transforms.ColorJitter(brightness=0.83, contrast=1.11, saturation=1.05, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
