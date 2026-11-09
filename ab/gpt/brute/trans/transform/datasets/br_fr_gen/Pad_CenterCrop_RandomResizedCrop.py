import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(102, 148, 118), padding_mode='reflect'),
    transforms.CenterCrop(size=32),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.9), ratio=(0.99, 2.94)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
