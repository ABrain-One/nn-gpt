import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.89), transforms.CenterCrop(size=28), transforms.RandomVerticalFlip(p=0.53), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])