import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(247, 131, 125), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.64),
    transforms.RandomPosterize(bits=4, p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
