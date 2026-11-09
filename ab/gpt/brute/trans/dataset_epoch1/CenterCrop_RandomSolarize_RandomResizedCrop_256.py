import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomSolarize(threshold=88, p=0.35),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.95), ratio=(1.2, 2.78)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
