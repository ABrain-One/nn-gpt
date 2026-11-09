import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.87), ratio=(1.14, 1.59)),
    transforms.RandomEqualize(p=0.46),
    transforms.RandomHorizontalFlip(p=0.41),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
