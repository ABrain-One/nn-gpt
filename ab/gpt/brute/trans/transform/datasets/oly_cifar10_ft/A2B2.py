import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(1.07, 1.89)),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.77),
    transforms.RandomAdjustSharpness(sharpness_factor=1.85, p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])