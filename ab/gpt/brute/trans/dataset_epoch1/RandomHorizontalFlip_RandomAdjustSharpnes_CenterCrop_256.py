import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.RandomAdjustSharpness(sharpness_factor=0.97, p=0.8),
    transforms.CenterCrop(size=25),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
