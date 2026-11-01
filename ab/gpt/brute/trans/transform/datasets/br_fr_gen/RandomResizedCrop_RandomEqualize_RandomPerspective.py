import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.92), ratio=(1.14, 2.78)),
    transforms.RandomEqualize(p=0.11),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
