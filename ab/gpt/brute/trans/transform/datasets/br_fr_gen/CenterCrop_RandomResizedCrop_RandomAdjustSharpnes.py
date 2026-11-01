import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.82), ratio=(1.19, 2.67)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
