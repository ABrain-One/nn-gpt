import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(255, 190, 17), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.88), ratio=(0.97, 1.79)),
    transforms.CenterCrop(size=32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
