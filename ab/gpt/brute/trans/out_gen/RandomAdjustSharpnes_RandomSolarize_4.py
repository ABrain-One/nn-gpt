import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.54, p=0.87),
    transforms.RandomSolarize(threshold=155, p=0.6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
