import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.83), ratio=(0.9, 1.81)),
    transforms.RandomCrop(size=30),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.35),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
