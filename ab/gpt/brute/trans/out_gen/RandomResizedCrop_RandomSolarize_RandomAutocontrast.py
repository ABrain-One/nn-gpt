import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.83), ratio=(0.99, 2.19)),
    transforms.RandomSolarize(threshold=152, p=0.7),
    transforms.RandomAutocontrast(p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
