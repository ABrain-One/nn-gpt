import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.82), ratio=(1.27, 1.89)),
    transforms.RandomHorizontalFlip(p=0.84),
    transforms.RandomAdjustSharpness(sharpness_factor=1.42, p=0.52),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
