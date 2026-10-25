import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=13, translate=(0.07, 0.17), scale=(1.03, 1.85), shear=(2.56, 9.26)),
    transforms.RandomPosterize(bits=4, p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
