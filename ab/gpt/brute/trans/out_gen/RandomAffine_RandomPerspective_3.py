import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.07, 0.05), scale=(1.0, 1.31), shear=(3.61, 6.94)),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
