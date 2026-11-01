import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomAdjustSharpness(sharpness_factor=1.14, p=0.52),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.86), ratio=(0.99, 1.73)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
