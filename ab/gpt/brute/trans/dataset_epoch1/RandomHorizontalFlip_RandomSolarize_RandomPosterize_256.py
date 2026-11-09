import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.35),
    transforms.RandomSolarize(threshold=84, p=0.6),
    transforms.RandomPosterize(bits=8, p=0.59),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
