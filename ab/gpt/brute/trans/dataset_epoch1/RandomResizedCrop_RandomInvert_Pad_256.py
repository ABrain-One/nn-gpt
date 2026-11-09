import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.85), ratio=(0.83, 1.41)),
    transforms.RandomInvert(p=0.65),
    transforms.Pad(padding=0, fill=(58, 140, 135), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
