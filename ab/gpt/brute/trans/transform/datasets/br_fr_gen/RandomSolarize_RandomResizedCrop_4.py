import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=158, p=0.6),
    transforms.RandomResizedCrop(size=32, scale=(0.5, 0.85), ratio=(0.76, 1.93)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
