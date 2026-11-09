import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.95), ratio=(0.84, 2.01)),
    transforms.RandomSolarize(threshold=153, p=0.25),
    transforms.RandomHorizontalFlip(p=0.66),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
