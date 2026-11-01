import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.95, contrast=0.96, saturation=1.0, hue=0.03),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.72),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
