import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.9), ratio=(0.95, 1.48)),
    transforms.Pad(padding=4, fill=(105, 10, 41), padding_mode='constant'),
    transforms.RandomInvert(p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
