import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(139, 62, 129), padding_mode='symmetric'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=0.6),
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.92), ratio=(0.8, 2.11)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])