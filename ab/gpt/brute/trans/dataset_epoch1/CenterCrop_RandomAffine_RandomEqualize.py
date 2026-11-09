import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=29),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.12), scale=(1.11, 1.83), shear=(3.32, 6.2)),
    transforms.RandomEqualize(p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
