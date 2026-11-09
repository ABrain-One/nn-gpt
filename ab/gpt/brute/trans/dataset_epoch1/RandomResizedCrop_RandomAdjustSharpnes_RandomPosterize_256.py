import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(0.77, 2.83)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.87, p=0.24),
    transforms.RandomPosterize(bits=4, p=0.16),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
