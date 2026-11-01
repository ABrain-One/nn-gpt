import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.58),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.73),
        transforms.RandomResizedCrop(size=64, scale=(0.69, 0.81), ratio=(0.95, 1.51)),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])