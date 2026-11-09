import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(137, 88, 75), padding_mode='constant'),
    transforms.RandomRotation(degrees=4),
    transforms.RandomPosterize(bits=8, p=0.46),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
