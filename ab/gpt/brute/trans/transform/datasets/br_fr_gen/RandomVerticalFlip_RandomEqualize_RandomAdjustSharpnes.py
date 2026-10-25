import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.75),
    transforms.RandomEqualize(p=0.35),
    transforms.RandomAdjustSharpness(sharpness_factor=1.44, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
