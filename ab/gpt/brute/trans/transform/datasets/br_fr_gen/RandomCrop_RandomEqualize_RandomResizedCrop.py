import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomEqualize(p=0.79),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.86), ratio=(0.93, 1.64)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
