import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(139, 105, 245), padding_mode='edge'), transforms.RandomAdjustSharpness(sharpness_factor=1.8, p=0.88), transforms.RandomInvert(p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])