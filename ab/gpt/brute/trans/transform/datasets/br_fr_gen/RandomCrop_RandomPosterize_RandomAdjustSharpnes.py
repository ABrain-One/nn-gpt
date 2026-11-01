import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomPosterize(bits=6, p=0.39),
    transforms.RandomAdjustSharpness(sharpness_factor=1.96, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
