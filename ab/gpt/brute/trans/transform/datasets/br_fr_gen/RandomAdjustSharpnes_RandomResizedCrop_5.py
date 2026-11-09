import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.59),
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.83), ratio=(1.08, 1.97)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
