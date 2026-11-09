import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomResizedCrop(size=32, scale=(0.65, 1.0), ratio=(1.08, 1.44)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
