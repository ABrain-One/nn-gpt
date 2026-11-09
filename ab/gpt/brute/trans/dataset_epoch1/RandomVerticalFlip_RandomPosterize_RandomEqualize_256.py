import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.63),
    transforms.RandomPosterize(bits=8, p=0.24),
    transforms.RandomEqualize(p=0.19),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
