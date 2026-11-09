import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.43),
    transforms.RandomPosterize(bits=5, p=0.35),
    transforms.RandomGrayscale(p=0.74),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
