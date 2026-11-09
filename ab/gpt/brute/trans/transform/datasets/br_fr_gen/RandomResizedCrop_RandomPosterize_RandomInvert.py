import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.86), ratio=(1.18, 2.73)),
    transforms.RandomPosterize(bits=5, p=0.81),
    transforms.RandomInvert(p=0.7),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
