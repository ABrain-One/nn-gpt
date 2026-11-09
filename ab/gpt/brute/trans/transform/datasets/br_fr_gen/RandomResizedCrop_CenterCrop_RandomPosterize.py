import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(1.22, 1.52)),
    transforms.CenterCrop(size=29),
    transforms.RandomPosterize(bits=5, p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
