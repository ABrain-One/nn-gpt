import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.56),
    transforms.RandomHorizontalFlip(p=0.67),
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.99), ratio=(1.28, 2.34)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
