import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.16),
    transforms.RandomAffine(degrees=25, translate=(0.14, 0.09), scale=(1.11, 1.91), shear=(3.67, 7.41)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
