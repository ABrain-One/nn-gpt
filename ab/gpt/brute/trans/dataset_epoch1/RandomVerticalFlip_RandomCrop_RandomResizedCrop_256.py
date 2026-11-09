import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.49),
    transforms.RandomCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.95), ratio=(1.08, 2.44)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
