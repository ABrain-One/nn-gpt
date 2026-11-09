import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomAffine(degrees=1, translate=(0.17, 0.18), scale=(1.06, 1.57), shear=(2.93, 5.8)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
