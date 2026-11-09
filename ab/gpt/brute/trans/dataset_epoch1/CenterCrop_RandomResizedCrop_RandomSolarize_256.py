import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.86), ratio=(0.87, 1.85)),
    transforms.RandomSolarize(threshold=148, p=0.46),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
