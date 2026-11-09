import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(173, 105, 35), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.97, p=0.62),
    transforms.RandomHorizontalFlip(p=0.72),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])