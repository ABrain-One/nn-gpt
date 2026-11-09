import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.94), ratio=(1.31, 2.66)),
    transforms.RandomCrop(size=32),
    transforms.RandomSolarize(threshold=241, p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
