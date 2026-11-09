import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.56),
    transforms.RandomEqualize(p=0.42),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.86), ratio=(1.0, 2.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
