import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.77), transforms.RandomEqualize(p=0.63), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.29, 0.2821, 0.2929], std=[0.2755, 0.2841, 0.2606])])