import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.87), ratio=(1.26, 1.47)),
    transforms.RandomPosterize(bits=8, p=0.84),
    transforms.RandomEqualize(p=0.45),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
