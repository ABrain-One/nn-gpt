import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomHorizontalFlip(p=0.51),
    transforms.RandomAffine(degrees=15, translate=(0.06, 0.17), scale=(1.07, 1.88), shear=(4.65, 7.35)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
