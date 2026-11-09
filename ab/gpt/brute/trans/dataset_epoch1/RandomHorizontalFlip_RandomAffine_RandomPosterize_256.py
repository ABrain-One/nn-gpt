import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.81),
    transforms.RandomAffine(degrees=17, translate=(0.1, 0.0), scale=(0.87, 1.2), shear=(0.46, 6.93)),
    transforms.RandomPosterize(bits=8, p=0.3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
