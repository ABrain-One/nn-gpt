import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(81, 134, 252), padding_mode='reflect'), transforms.RandomAffine(degrees=13, translate=(0.05, 0.17), scale=(0.99, 1.99), shear=(2.23, 7.5)), transforms.RandomPosterize(bits=7, p=0.83), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])