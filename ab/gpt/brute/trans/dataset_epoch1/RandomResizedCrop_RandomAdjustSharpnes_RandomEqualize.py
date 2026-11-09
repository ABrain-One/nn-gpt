import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.94), ratio=(0.89, 2.32)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.63, p=0.88),
    transforms.RandomEqualize(p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
