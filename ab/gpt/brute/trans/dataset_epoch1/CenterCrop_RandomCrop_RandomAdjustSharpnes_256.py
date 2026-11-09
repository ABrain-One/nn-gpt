import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomCrop(size=32),
    transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.47),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
