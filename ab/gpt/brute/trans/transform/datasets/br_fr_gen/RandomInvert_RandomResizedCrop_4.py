import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.26),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.97), ratio=(1.03, 1.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
