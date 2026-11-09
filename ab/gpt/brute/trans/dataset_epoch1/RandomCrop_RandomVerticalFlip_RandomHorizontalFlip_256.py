import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomVerticalFlip(p=0.22),
    transforms.RandomHorizontalFlip(p=0.26),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
