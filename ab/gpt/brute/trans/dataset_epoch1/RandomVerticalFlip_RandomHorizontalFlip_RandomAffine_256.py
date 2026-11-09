import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.68),
    transforms.RandomHorizontalFlip(p=0.13),
    transforms.RandomAffine(degrees=9, translate=(0.09, 0.09), scale=(1.1, 1.41), shear=(2.17, 9.93)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
