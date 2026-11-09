import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.98), ratio=(0.97, 2.75)),
    transforms.CenterCrop(size=28),
    transforms.RandomSolarize(threshold=26, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
