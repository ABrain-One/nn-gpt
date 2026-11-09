import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.56),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.94), ratio=(1.04, 1.89)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
