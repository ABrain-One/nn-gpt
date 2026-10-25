import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.55, 1.0), ratio=(0.79, 2.29)),
    transforms.RandomHorizontalFlip(p=0.27),
    transforms.RandomEqualize(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
