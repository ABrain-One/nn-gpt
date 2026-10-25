import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(131, 125, 163), padding_mode='constant'),
    transforms.RandomRotation(degrees=0),
    transforms.RandomPosterize(bits=4, p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])