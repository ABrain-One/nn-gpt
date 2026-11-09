import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(151, 86, 127), padding_mode='constant'),
    transforms.RandomPosterize(bits=7, p=0.57),
    transforms.RandomVerticalFlip(p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])