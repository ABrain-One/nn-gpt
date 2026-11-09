import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(16, 141, 111), padding_mode='edge'),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.95), ratio=(1.18, 2.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
