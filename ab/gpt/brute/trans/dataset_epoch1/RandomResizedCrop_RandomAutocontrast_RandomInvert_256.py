import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.81), ratio=(0.95, 1.5)),
    transforms.RandomAutocontrast(p=0.76),
    transforms.RandomInvert(p=0.81),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
