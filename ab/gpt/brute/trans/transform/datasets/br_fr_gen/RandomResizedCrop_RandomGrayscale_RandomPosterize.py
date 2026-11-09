import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.81), ratio=(0.83, 1.65)),
    transforms.RandomGrayscale(p=0.66),
    transforms.RandomPosterize(bits=7, p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
