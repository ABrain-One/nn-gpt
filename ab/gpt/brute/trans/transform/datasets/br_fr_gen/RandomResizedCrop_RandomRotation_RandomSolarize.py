import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.74, 0.83), ratio=(1.0, 1.94)),
    transforms.RandomRotation(degrees=1),
    transforms.RandomSolarize(threshold=143, p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
