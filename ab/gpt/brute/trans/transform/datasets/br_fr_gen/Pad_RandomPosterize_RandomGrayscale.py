import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(160, 240, 152), padding_mode='edge'),
    transforms.RandomPosterize(bits=6, p=0.16),
    transforms.RandomGrayscale(p=0.72),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
