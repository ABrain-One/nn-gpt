import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.36),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.66),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.83), ratio=(0.79, 2.3)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
