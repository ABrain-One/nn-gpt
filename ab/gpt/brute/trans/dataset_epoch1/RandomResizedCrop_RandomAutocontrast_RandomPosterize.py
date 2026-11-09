import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.82), ratio=(0.84, 2.94)),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomPosterize(bits=4, p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
