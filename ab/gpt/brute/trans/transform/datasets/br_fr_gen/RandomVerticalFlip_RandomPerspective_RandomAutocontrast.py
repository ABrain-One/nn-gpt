import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.65),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.83),
    transforms.RandomAutocontrast(p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
