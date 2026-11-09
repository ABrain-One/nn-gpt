import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.95), ratio=(1.1, 2.55)),
    transforms.RandomSolarize(threshold=184, p=0.24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
