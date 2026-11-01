import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=1, translate=(0.18, 0.19), scale=(1.02, 1.26), shear=(1.08, 8.98)),
    transforms.RandomPosterize(bits=7, p=0.46),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
