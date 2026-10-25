import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.18, p=0.43),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.93), ratio=(0.79, 2.2)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
