import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=4, p=0.81),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.84), ratio=(0.94, 2.47)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
