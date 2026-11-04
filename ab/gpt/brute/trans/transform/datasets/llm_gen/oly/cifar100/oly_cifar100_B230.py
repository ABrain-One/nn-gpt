import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.64), transforms.Grayscale(num_output_channels=3), transforms.RandomCrop(27), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])