import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.68),
    transforms.RandomPosterize(bits=4, p=0.37),
    transforms.RandomAdjustSharpness(sharpness_factor=1.72, p=0.13),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
