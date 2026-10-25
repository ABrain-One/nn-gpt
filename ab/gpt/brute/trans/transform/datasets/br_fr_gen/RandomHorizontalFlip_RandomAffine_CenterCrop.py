import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.77),
    transforms.RandomAffine(degrees=18, translate=(0.1, 0.09), scale=(0.95, 1.26), shear=(0.47, 8.56)),
    transforms.CenterCrop(size=31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
