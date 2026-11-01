import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.55), transforms.RandomSolarize(threshold=125, p=0.46), transforms.RandomAdjustSharpness(sharpness_factor=1.61, p=0.7), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])