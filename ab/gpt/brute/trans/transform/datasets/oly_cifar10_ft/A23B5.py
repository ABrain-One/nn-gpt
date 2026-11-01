import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(173, 142, 81), padding_mode='edge'),
    transforms.RandomPosterize(bits=8, p=0.85),
    transforms.RandomAdjustSharpness(sharpness_factor=0.88, p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])