import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=26), transforms.RandomPerspective(distortion_scale=0.26, p=0.71), transforms.RandomVerticalFlip(p=0.82), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])