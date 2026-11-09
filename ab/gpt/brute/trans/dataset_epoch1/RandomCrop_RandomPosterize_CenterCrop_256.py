import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomPosterize(bits=6, p=0.61),
    transforms.CenterCrop(size=28),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
