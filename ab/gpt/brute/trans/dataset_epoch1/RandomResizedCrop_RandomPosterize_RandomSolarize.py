import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.82), ratio=(1.05, 2.22)),
    transforms.RandomPosterize(bits=8, p=0.65),
    transforms.RandomSolarize(threshold=41, p=0.44),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
