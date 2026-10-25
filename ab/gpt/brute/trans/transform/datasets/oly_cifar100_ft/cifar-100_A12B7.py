import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.19), transforms.RandomAffine(degrees=25, translate=(0.14, 0.03), scale=(1.03, 1.72), shear=(3.47, 8.05)), transforms.Pad(padding=4, fill=(102, 7, 181), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])