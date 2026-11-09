import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.14, p=0.15),
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.83), ratio=(0.96, 2.6)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
