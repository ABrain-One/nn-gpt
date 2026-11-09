import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomPosterize(bits=4, p=0.47),
    transforms.RandomAffine(degrees=9, translate=(0.0, 0.17), scale=(1.03, 1.73), shear=(4.4, 5.49)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
