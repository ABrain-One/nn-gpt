import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.81),
    transforms.RandomAdjustSharpness(sharpness_factor=1.27, p=0.14),
    transforms.RandomCrop(size=31),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
