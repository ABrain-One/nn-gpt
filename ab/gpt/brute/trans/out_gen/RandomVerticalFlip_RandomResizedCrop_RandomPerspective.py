import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.39),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.81), ratio=(0.99, 2.8)),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
