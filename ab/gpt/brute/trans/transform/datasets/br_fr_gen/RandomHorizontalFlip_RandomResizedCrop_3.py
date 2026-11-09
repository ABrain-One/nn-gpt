import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.69),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.8), ratio=(0.88, 1.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
