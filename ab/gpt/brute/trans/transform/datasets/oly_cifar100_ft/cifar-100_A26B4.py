import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.Grayscale(num_output_channels=3), transforms.RandomHorizontalFlip(p=0.51), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean=0.487, std=0.227)])