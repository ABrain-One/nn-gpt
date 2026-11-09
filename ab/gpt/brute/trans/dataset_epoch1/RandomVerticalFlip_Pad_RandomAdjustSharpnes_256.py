import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.44),
    transforms.Pad(padding=2, fill=(52, 105, 205), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.53, p=0.61),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
