import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.96), ratio=(1.02, 1.72)),
    transforms.RandomSolarize(threshold=213, p=0.78),
    transforms.RandomPosterize(bits=6, p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
