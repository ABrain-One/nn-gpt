import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomEqualize(p=0.65),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.96), ratio=(1.19, 1.34)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
