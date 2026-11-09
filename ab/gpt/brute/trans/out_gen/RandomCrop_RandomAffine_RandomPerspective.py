import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.RandomAffine(degrees=26, translate=(0.09, 0.09), scale=(1.0, 1.84), shear=(1.51, 8.06)),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
