import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.34),
    transforms.RandomAutocontrast(p=0.31),
    transforms.RandomSolarize(threshold=34, p=0.67),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
