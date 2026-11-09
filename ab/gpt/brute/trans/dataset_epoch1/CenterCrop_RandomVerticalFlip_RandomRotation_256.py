import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomVerticalFlip(p=0.29),
    transforms.RandomRotation(degrees=20),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
