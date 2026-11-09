import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.RandomEqualize(p=0.24),
    transforms.RandomAffine(degrees=26, translate=(0.16, 0.11), scale=(0.99, 1.3), shear=(0.53, 6.22)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
