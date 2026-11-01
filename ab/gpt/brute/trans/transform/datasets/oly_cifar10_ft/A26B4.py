import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(226, 56, 165), padding_mode='edge'),
    transforms.RandomGrayscale(p=0.35),
    transforms.RandomPosterize(bits=4, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm))
]