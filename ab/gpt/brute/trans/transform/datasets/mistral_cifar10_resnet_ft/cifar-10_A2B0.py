import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomEqualize(p=0.76), transforms.RandomAdjustSharpness(sharpness_factor=1.93, p=0.85), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])