import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(165, 82, 82), padding_mode='edge'), transforms.RandomAdjustSharpness(sharpness_factor=0.51, p=0.77), transforms.RandomPosterize(bits=7, p=0.16), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])