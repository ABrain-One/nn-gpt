import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(254, 162, 90), padding_mode='constant'), transforms.RandomPosterize(bits=7, p=0.38), transforms.CenterCrop(size=30), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])