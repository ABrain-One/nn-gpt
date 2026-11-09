import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.94), ratio=(1.18, 1.44)),
    transforms.RandomInvert(p=0.15),
    transforms.RandomPerspective(distortion_scale=0.24, p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
