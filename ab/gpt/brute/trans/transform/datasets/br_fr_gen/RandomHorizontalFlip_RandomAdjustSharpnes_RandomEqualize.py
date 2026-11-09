import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.61),
    transforms.RandomAdjustSharpness(sharpness_factor=1.79, p=0.59),
    transforms.RandomEqualize(p=0.56),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
