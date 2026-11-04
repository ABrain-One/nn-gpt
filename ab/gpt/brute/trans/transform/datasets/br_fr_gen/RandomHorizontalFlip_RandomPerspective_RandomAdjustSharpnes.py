import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.87),
    transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
