import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomAutocontrast(p=0.41),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.94), ratio=(1.31, 2.26)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
