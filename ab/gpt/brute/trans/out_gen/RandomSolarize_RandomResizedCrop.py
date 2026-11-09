import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=61, p=0.69),
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.98), ratio=(1.15, 2.21)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
