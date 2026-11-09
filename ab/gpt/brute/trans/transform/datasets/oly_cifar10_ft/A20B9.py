import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(197, 85, 249), padding_mode='edge'),
    transforms.RandomPosterize(bits=8, p=0.34),
    transforms.RandomAdjustSharpness(sharpness_factor=1.12, p=0.5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])