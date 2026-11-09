import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.36), transforms.Pad(padding=0, fill=(198, 7, 128), padding_mode='constant'), transforms.RandomAdjustSharpness(sharpness_factor=1.89, p=0.62), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])