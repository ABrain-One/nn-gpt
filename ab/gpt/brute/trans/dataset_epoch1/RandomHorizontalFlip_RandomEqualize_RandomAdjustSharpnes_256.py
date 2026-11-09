import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.72),
    transforms.RandomEqualize(p=0.46),
    transforms.RandomAdjustSharpness(sharpness_factor=1.46, p=0.48),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
