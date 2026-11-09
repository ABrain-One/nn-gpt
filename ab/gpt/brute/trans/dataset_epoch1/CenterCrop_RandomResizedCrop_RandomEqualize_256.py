import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.92), ratio=(1.19, 2.42)),
    transforms.RandomEqualize(p=0.28),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
