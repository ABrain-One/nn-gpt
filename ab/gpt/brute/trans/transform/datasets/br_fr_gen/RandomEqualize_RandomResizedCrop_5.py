import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.67),
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.97), ratio=(1.26, 2.01)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
