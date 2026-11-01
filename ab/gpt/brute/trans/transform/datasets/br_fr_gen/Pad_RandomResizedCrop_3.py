import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(80, 0, 112), padding_mode='symmetric'),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.84), ratio=(1.23, 1.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
