import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 0.8)), transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5.0, 5.0)), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.GaussianBlur(3, sigma=0.5, p=0.5), transforms.Pad(5, fill=(224, 248, 148), padding_mode='constant'), transforms.Resize((64, 64), interpolation=transforms.Interpolation.BICUBIC), transforms.ToTensor(), transforms.Normalize(*norm)])