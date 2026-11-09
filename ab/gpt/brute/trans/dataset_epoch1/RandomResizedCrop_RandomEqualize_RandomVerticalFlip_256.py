import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.95), ratio=(1.09, 1.44)),
    transforms.RandomEqualize(p=0.21),
    transforms.RandomVerticalFlip(p=0.21),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
