import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.85), ratio=(0.84, 1.89)),
    transforms.RandomGrayscale(p=0.85),
    transforms.RandomAdjustSharpness(sharpness_factor=1.73, p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
