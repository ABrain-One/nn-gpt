import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.34),
    transforms.CenterCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.86), ratio=(1.06, 1.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
