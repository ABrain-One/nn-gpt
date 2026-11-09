import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(175, 195, 128), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.72, p=0.12),
    transforms.RandomHorizontalFlip(p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])