import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.88), ratio=(0.88, 1.53)),
    transforms.RandomPosterize(bits=7, p=0.32),
    transforms.RandomRotation(degrees=0),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
