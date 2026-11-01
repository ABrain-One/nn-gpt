import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.87), ratio=(1.27, 1.76)),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.36),
    transforms.RandomPosterize(bits=8, p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
