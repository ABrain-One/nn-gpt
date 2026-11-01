import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(131, 69, 185), padding_mode='edge'),
    transforms.RandomVerticalFlip(p=0.82),
    transforms.RandomPosterize(bits=6, p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])