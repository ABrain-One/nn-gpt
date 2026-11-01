import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(213, 181, 213), padding_mode='symmetric'), transforms.RandomAdjustSharpness(sharpness_factor=1.98, p=0.61), transforms.CenterCrop(size=30), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])