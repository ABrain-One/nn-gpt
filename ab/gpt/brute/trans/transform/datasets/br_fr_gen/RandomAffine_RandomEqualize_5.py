import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.09, 0.06), scale=(1.03, 1.34), shear=(0.82, 6.73)),
    transforms.RandomEqualize(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
