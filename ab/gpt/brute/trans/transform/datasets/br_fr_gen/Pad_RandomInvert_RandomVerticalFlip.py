import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(100, 232, 63), padding_mode='edge'),
    transforms.RandomInvert(p=0.41),
    transforms.RandomVerticalFlip(p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
