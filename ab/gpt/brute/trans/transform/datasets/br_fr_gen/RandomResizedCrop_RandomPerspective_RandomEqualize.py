import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.86), ratio=(0.9, 2.43)),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.86),
    transforms.RandomEqualize(p=0.25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
