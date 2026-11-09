import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 1.0), ratio=(1.03, 1.62)),
    transforms.RandomEqualize(p=0.65),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
