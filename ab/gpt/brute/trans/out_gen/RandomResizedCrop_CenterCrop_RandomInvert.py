import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.86), ratio=(1.07, 1.84)),
    transforms.CenterCrop(size=29),
    transforms.RandomInvert(p=0.59),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
