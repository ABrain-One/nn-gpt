import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(233, 23, 117), padding_mode='constant'),
    transforms.RandomRotation(degrees=14),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.9), ratio=(1.02, 1.34)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
