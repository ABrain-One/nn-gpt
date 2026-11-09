import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.23),
    transforms.RandomPosterize(bits=5, p=0.71),
    transforms.RandomVerticalFlip(p=0.51),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
