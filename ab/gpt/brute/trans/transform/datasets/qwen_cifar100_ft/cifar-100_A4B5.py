import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.Pad(padding=3, fill=(255, 103, 171), padding_mode='edge'), transforms.RandomPosterize(bits=4, p=0.82), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])