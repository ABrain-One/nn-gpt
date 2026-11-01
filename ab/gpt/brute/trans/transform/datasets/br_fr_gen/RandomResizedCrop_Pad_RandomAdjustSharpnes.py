import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.84), ratio=(1.19, 1.59)),
    transforms.Pad(padding=1, fill=(158, 163, 1), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.71, p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
