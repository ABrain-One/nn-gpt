import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.RandomSolarize(threshold=162, p=0.3), transforms.Pad(padding=4, fill=(181, 219, 46), padding_mode='edge'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])