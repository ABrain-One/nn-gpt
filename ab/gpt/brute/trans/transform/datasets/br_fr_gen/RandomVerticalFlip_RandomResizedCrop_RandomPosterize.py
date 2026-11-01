import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.35),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.98), ratio=(1.29, 2.58)),
    transforms.RandomPosterize(bits=4, p=0.89),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
