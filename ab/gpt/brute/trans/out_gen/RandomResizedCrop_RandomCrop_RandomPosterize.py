import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.89), ratio=(1.01, 2.53)),
    transforms.RandomCrop(size=25),
    transforms.RandomPosterize(bits=5, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
