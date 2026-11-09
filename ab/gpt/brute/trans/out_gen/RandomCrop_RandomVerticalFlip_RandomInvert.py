import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomInvert(p=0.37),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
