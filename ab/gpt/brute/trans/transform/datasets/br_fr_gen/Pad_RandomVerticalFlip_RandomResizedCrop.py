import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(13, 201, 32), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.66),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.81), ratio=(0.78, 2.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
