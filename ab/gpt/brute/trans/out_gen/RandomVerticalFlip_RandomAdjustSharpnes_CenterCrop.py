import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.9),
    transforms.RandomAdjustSharpness(sharpness_factor=1.35, p=0.39),
    transforms.CenterCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
