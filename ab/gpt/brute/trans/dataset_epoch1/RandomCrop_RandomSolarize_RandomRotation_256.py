import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomSolarize(threshold=155, p=0.5),
    transforms.RandomRotation(degrees=9),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
