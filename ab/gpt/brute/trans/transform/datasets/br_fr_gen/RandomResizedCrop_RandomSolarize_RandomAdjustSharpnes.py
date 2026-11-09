import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.83), ratio=(1.08, 2.92)),
    transforms.RandomSolarize(threshold=38, p=0.44),
    transforms.RandomAdjustSharpness(sharpness_factor=1.75, p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
