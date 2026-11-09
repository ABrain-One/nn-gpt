import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPosterize(bits=5, p=0.79),
    transforms.RandomAffine(degrees=27, translate=(0.18, 0.09), scale=(0.97, 1.66), shear=(0.42, 9.75)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
