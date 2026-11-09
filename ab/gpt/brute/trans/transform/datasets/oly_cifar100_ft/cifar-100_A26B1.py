import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(174, 146, 74), padding_mode='reflect'), transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(0.97, 2.94)), transforms.RandomAdjustSharpness(sharpness_factor=1.34, p=0.38), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])