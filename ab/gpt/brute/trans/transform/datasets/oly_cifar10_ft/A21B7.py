import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(1, 12, 199), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.57, p=0.7),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])