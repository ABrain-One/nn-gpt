import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(148, 202, 171), padding_mode='edge'),
    transforms.CenterCrop(size=32),
    transforms.RandomAdjustSharpness(sharpness_factor=0.76, p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
