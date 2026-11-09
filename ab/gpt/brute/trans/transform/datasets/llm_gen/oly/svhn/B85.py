import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.5, 3.0)), transforms.ColorJitter(0.4, 0.4, 0.4, 0.2), transforms.RandomEqualize(p=0.8), transforms.RandomSolarize(threshold=130, p=0.2), transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5), transforms.RandomVerticalFlip(p=0.2), transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(*norm)])