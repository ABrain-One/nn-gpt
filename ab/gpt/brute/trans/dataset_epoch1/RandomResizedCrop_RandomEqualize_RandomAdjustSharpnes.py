import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.98), ratio=(0.86, 1.47)),
    transforms.RandomEqualize(p=0.12),
    transforms.RandomAdjustSharpness(sharpness_factor=1.87, p=0.51),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
