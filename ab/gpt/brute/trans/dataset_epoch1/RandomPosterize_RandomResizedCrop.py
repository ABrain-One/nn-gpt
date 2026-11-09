import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=4, p=0.64),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.89), ratio=(1.15, 2.16)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
