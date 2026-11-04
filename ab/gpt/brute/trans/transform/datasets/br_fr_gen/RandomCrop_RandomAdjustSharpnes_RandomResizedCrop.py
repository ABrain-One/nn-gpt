import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomAdjustSharpness(sharpness_factor=1.39, p=0.84),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.98), ratio=(1.03, 2.59)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
