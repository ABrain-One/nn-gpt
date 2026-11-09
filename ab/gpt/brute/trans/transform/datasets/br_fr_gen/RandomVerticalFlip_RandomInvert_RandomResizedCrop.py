import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.27),
    transforms.RandomInvert(p=0.2),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.8), ratio=(0.92, 2.36)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
