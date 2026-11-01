import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.94), ratio=(0.75, 2.71)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.56, p=0.21),
    transforms.RandomGrayscale(p=0.41),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
