import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.11, p=0.75),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.81), ratio=(0.8, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
