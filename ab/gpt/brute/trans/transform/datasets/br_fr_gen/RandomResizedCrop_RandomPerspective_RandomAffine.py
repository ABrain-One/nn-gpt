import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.91), ratio=(1.1, 1.51)),
    transforms.RandomPerspective(distortion_scale=0.29, p=0.57),
    transforms.RandomAffine(degrees=21, translate=(0.01, 0.09), scale=(1.13, 1.66), shear=(2.09, 8.84)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
