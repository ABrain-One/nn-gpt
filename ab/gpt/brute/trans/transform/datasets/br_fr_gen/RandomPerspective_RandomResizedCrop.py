import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.17, p=0.12),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.82), ratio=(1.08, 2.24)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
