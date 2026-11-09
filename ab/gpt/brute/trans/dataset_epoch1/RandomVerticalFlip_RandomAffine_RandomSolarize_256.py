import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.23),
    transforms.RandomAffine(degrees=5, translate=(0.06, 0.01), scale=(1.15, 1.45), shear=(1.58, 5.78)),
    transforms.RandomSolarize(threshold=167, p=0.84),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
