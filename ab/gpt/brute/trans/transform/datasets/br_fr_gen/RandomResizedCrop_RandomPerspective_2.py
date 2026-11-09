import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.93), ratio=(1.29, 2.7)),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
