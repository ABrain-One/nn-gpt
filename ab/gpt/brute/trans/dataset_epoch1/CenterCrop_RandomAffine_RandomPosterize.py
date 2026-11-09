import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomAffine(degrees=23, translate=(0.02, 0.0), scale=(1.0, 1.78), shear=(4.31, 5.16)),
    transforms.RandomPosterize(bits=4, p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
