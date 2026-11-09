import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.82), ratio=(1.24, 1.93)),
    transforms.RandomInvert(p=0.7),
    transforms.RandomEqualize(p=0.52),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
