import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.94), ratio=(0.91, 2.56)),
    transforms.Pad(padding=3, fill=(63, 248, 67), padding_mode='constant'),
    transforms.RandomAutocontrast(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
