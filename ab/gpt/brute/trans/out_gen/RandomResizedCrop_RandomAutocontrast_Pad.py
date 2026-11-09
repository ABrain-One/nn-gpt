import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.8), ratio=(1.16, 1.49)),
    transforms.RandomAutocontrast(p=0.38),
    transforms.Pad(padding=5, fill=(0, 255, 193), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
