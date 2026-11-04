import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=1.84, p=0.67), transforms.Pad(padding=5, fill=(195, 216, 105), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])