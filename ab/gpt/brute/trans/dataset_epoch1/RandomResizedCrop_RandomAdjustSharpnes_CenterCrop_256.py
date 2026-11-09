import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.81), ratio=(1.14, 2.54)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.92, p=0.33),
    transforms.CenterCrop(size=29),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
