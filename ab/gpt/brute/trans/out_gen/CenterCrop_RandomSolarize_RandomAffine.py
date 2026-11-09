import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomSolarize(threshold=36, p=0.68),
    transforms.RandomAffine(degrees=12, translate=(0.17, 0.18), scale=(1.15, 1.91), shear=(4.29, 5.91)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
