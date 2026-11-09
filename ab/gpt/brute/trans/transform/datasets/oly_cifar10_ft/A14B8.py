import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(102, 109, 177), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.71, p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])