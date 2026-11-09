import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomSolarize(threshold=236, p=0.45),
    transforms.RandomAutocontrast(p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
