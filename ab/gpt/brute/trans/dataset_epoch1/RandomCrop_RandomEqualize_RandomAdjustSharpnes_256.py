import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=28),
    transforms.RandomEqualize(p=0.36),
    transforms.RandomAdjustSharpness(sharpness_factor=0.62, p=0.34),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
