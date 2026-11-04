import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(60, 254, 161), padding_mode='edge'),
    transforms.RandomVerticalFlip(p=0.62),
    transforms.RandomInvert(p=0.7),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
