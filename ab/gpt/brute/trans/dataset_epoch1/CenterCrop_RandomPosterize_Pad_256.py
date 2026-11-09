import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomPosterize(bits=7, p=0.71),
    transforms.Pad(padding=5, fill=(203, 135, 23), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
