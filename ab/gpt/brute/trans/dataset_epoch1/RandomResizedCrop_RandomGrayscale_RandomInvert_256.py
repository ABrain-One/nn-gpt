import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.84), ratio=(0.98, 2.6)),
    transforms.RandomGrayscale(p=0.55),
    transforms.RandomInvert(p=0.85),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
