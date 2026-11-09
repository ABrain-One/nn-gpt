import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(96, 188, 228), padding_mode='reflect'),
    transforms.RandomAutocontrast(p=0.14),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.87), ratio=(1.23, 1.68)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
