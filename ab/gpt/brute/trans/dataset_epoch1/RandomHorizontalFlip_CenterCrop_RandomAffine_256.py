import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.31),
    transforms.CenterCrop(size=32),
    transforms.RandomAffine(degrees=19, translate=(0.08, 0.06), scale=(0.84, 1.46), shear=(0.99, 7.03)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
