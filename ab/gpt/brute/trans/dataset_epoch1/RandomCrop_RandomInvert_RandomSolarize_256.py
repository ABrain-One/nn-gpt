import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomInvert(p=0.18),
    transforms.RandomSolarize(threshold=64, p=0.83),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
