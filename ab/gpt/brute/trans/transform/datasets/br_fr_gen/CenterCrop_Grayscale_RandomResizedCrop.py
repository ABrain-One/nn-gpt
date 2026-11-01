import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.83), ratio=(0.88, 2.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
