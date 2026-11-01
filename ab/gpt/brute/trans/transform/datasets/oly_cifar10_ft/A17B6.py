import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(43, 104, 101), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.92, p=0.69),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])