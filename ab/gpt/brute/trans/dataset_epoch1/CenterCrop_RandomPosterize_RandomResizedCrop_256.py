import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomPosterize(bits=7, p=0.25),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.84), ratio=(0.97, 3.0)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
