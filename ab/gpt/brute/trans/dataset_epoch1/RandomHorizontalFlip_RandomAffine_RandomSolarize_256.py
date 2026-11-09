import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.73),
    transforms.RandomAffine(degrees=8, translate=(0.15, 0.01), scale=(0.99, 1.49), shear=(0.49, 6.02)),
    transforms.RandomSolarize(threshold=247, p=0.54),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
