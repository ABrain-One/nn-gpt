import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(155, 158, 78), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.87),
    transforms.RandomVerticalFlip(p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])