import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.84),
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.98), ratio=(1.11, 1.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
