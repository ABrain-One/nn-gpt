import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.13),
    transforms.RandomPosterize(bits=6, p=0.48),
    transforms.RandomSolarize(threshold=36, p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
