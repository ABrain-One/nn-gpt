import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(132, 68, 62), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.8),
    transforms.RandomPosterize(bits=7, p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(**norm)
])