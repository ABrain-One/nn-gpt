import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.95), ratio=(1.16, 1.35)),
    transforms.RandomPosterize(bits=6, p=0.75),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
