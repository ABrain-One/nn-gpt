import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.57),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
