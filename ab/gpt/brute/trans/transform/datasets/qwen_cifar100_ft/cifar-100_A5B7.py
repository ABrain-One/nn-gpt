import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.63), transforms.CenterCrop(size=25), transforms.RandomPosterize(bits=8, p=0.51), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])