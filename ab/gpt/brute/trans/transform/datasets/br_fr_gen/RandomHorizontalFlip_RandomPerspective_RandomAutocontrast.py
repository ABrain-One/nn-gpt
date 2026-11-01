import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.87),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.54),
    transforms.RandomAutocontrast(p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
