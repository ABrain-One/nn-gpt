import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.39),
    transforms.RandomAffine(degrees=22, translate=(0.16, 0.14), scale=(1.19, 1.45), shear=(1.87, 8.65)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
