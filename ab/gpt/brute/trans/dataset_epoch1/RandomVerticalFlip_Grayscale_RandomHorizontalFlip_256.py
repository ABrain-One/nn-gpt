import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.27),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
