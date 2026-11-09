import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.9), ratio=(1.12, 2.6)),
    transforms.RandomPosterize(bits=6, p=0.83),
    transforms.RandomGrayscale(p=0.68),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
