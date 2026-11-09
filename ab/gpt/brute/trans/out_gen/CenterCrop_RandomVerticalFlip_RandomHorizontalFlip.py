import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomVerticalFlip(p=0.75),
    transforms.RandomHorizontalFlip(p=0.26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
