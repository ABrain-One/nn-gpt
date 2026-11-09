import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.94), ratio=(0.97, 2.02)),
    transforms.RandomSolarize(threshold=57, p=0.43),
    transforms.RandomCrop(size=28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
