import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomAffine(degrees=18, translate=(0.02, 0.06), scale=(0.97, 1.34), shear=(2.61, 5.49)),
    transforms.RandomRotation(degrees=1),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
