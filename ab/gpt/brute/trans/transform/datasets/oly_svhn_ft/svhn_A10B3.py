import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=29), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.223, 0.234, 0.22], std=[1.86, 1.82, 1.82])])