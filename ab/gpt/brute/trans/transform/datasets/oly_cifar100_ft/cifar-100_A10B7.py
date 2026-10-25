import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.22), transforms.RandomPerspective(distortion_scale=0.13, p=0.26), transforms.RandomEqualize(p=0.73), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])