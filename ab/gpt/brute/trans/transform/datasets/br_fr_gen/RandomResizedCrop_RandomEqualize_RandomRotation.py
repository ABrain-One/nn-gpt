import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.91), ratio=(0.97, 2.89)),
    transforms.RandomEqualize(p=0.32),
    transforms.RandomRotation(degrees=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
