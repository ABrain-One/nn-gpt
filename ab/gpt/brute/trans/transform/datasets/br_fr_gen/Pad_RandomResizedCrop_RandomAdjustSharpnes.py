import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(122, 135, 239), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.88), ratio=(1.04, 1.77)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.72, p=0.6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
