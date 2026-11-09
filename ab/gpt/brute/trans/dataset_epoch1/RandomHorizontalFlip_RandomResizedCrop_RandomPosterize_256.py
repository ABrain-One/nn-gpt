import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.RandomResizedCrop(size=32, scale=(0.63, 1.0), ratio=(1.16, 2.0)),
    transforms.RandomPosterize(bits=4, p=0.39),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
