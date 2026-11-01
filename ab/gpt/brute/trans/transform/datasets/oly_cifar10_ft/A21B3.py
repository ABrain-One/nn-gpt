import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(168, 195, 163), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.14, p=0.62),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(1.03, 1.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])