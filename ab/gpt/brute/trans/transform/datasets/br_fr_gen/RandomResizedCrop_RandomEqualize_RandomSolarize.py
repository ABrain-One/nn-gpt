import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.93), ratio=(0.8, 2.22)),
    transforms.RandomEqualize(p=0.89),
    transforms.RandomSolarize(threshold=132, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
