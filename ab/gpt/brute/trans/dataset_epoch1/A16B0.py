import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(13, 39), padding_mode='edge'),
    transforms.RandomPosterize(bits=7, p=0.43),
    transforms.RandomAdjustSharpness(sharpness_factor=1.06, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])