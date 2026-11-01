import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(234, 40, 167), padding_mode='reflect'),
    transforms.RandomPosterize(bits=8, p=0.55),
    transforms.RandomHorizontalFlip(p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])