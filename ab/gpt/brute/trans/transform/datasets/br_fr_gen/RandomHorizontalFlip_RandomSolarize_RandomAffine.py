import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.62),
    transforms.RandomSolarize(threshold=216, p=0.84),
    transforms.RandomAffine(degrees=22, translate=(0.16, 0.04), scale=(1.09, 1.95), shear=(2.45, 9.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
