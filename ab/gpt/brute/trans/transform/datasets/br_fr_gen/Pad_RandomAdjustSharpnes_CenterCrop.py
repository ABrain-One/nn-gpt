import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(114, 42, 195), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.77, p=0.44),
    transforms.CenterCrop(size=28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
