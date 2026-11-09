import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.96), ratio=(0.91, 2.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
