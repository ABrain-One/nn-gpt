import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.13),
    transforms.RandomAdjustSharpness(sharpness_factor=1.89, p=0.37),
    transforms.RandomSolarize(threshold=166, p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
