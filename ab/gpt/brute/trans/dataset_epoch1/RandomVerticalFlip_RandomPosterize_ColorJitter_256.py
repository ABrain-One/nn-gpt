import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.61),
    transforms.RandomPosterize(bits=8, p=0.49),
    transforms.ColorJitter(brightness=1.08, contrast=0.98, saturation=1.13, hue=0.02),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
