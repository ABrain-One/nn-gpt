import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.CenterCrop(size=24),
    transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
