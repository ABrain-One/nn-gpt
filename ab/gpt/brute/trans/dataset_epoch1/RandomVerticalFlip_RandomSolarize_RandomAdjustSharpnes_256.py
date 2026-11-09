import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.32),
    transforms.RandomSolarize(threshold=141, p=0.24),
    transforms.RandomAdjustSharpness(sharpness_factor=1.78, p=0.7),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
