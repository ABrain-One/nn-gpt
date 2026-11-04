import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(201, 131, 30), padding_mode='edge'),
    transforms.RandomCrop(size=27),
    transforms.RandomAdjustSharpness(sharpness_factor=1.57, p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
