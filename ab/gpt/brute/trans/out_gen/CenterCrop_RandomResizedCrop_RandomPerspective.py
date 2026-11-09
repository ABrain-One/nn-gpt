import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.92), ratio=(1.05, 2.71)),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
