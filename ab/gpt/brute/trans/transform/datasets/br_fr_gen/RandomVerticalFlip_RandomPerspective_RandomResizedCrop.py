import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.82),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.44),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.92), ratio=(1.32, 1.75)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
