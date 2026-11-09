import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomAdjustSharpness(sharpness_factor=1.36, p=0.22),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
