import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomSolarize(threshold=44, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
