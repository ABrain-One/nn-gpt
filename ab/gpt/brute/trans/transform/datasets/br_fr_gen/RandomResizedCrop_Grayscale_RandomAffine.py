import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.81), ratio=(1.26, 1.69)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=13, translate=(0.09, 0.16), scale=(0.93, 1.9), shear=(4.34, 8.81)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
