import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.7),
    transforms.RandomPosterize(bits=8, p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
