import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=141, p=0.31),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.91), ratio=(1.24, 3.0)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
