import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.85), ratio=(1.24, 1.72)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.47, p=0.46),
    transforms.RandomSolarize(threshold=252, p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
