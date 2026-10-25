import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=19, translate=(0.2, 0.04), scale=(0.81, 1.27), shear=(3.01, 5.26)),
    transforms.RandomVerticalFlip(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
