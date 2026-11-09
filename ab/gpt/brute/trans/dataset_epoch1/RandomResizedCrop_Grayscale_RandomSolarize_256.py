import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.84), ratio=(0.88, 1.59)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomSolarize(threshold=152, p=0.45),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
