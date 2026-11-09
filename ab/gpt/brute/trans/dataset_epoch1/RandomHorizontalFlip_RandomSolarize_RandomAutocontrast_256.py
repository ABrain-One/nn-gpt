import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.15),
    transforms.RandomSolarize(threshold=124, p=0.67),
    transforms.RandomAutocontrast(p=0.21),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
