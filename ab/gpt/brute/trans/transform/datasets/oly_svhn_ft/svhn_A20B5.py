import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.4468, 0.445, 0.4404], std=[0.281, 0.2818, 0.282])])