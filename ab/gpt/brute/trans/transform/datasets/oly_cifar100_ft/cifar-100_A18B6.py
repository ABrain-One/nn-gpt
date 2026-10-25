import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(189, 135, 83), padding_mode='edge'), transforms.RandomAdjustSharpness(sharpness_factor=1.7, p=0.88), transforms.RandomPerspective(distortion_scale=0.17, p=0.13), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])