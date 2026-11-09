import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.94), ratio=(1.32, 2.09)),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.33),
    transforms.RandomVerticalFlip(p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
