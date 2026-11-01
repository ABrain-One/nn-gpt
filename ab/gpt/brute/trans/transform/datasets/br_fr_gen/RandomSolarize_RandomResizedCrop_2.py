import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=101, p=0.53),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.92), ratio=(1.02, 2.11)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
