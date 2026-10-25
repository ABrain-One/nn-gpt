import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.75),
    transforms.RandomPosterize(bits=4, p=0.58),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.97), ratio=(0.91, 2.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
