import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.Pad(padding=4, fill=(177, 80, 133), padding_mode='symmetric'), transforms.RandomAdjustSharpness(sharpness_factor=1.88, p=0.62), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])