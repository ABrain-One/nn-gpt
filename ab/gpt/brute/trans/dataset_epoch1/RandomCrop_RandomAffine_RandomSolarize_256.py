import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomAffine(degrees=22, translate=(0.0, 0.1), scale=(0.89, 1.47), shear=(4.52, 5.59)),
    transforms.RandomSolarize(threshold=244, p=0.63),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
