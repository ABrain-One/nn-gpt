import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.49),
    transforms.RandomAffine(degrees=20, translate=(0.12, 0.11), scale=(0.96, 1.44), shear=(0.67, 6.61)),
    transforms.RandomGrayscale(p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
