import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomAffine(degrees=17, translate=(0.19, 0.17), scale=(0.86, 1.7), shear=(4.04, 6.66)),
    transforms.RandomGrayscale(p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
