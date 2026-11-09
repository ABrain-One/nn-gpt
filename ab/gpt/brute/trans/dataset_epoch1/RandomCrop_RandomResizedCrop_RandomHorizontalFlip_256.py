import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.81), ratio=(1.15, 2.84)),
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
