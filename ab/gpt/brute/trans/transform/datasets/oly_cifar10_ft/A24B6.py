import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.87), ratio=(1.11, 1.43)),
    transforms.RandomPosterize(bits=8, p=0.17),
    transforms.RandomAdjustSharpness(sharpness_factor=1.72, p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])