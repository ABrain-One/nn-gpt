import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.92), ratio=(1.17, 2.54)),
    transforms.RandomAutocontrast(p=0.12),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.63),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
