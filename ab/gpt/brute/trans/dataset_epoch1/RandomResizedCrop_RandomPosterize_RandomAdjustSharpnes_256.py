import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.95), ratio=(0.84, 1.62)),
    transforms.RandomPosterize(bits=4, p=0.3),
    transforms.RandomAdjustSharpness(sharpness_factor=1.46, p=0.3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
