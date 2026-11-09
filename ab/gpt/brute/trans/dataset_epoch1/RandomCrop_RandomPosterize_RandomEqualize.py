import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.RandomPosterize(bits=7, p=0.36),
    transforms.RandomEqualize(p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
