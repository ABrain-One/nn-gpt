import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.83), ratio=(0.98, 2.41)),
    transforms.CenterCrop(size=32),
    transforms.RandomAdjustSharpness(sharpness_factor=0.85, p=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
