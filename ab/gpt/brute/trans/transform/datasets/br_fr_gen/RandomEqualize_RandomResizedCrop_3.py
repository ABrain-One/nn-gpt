import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.89),
    transforms.RandomResizedCrop(size=32, scale=(0.79, 1.0), ratio=(0.81, 2.64)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
