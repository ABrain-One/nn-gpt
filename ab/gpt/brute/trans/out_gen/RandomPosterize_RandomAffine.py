import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=8, p=0.7),
    transforms.RandomAffine(degrees=23, translate=(0.06, 0.04), scale=(1.18, 1.72), shear=(1.54, 5.41)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
