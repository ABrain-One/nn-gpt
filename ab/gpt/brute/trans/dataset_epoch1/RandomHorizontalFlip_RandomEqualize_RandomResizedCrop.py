import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.19),
    transforms.RandomEqualize(p=0.84),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.91), ratio=(0.78, 1.75)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
