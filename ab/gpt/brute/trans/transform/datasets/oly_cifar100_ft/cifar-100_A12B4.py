import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomResizedCrop(size=32, scale=(0.63, 0.82), ratio=(1.19, 2.3)), transforms.RandomAdjustSharpness(sharpness_factor=0.82, p=0.45), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize((0.481, 0.492, 0.915), (0.238, 0.235, 0.231))])