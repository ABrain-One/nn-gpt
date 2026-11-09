import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=51, p=0.62),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.87), ratio=(1.29, 2.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
