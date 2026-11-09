import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.97), ratio=(1.04, 2.01)),
    transforms.RandomEqualize(p=0.11),
    transforms.RandomInvert(p=0.79),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
