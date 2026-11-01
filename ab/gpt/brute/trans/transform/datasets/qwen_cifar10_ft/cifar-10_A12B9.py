import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(180, 212, 190), padding_mode='edge'), transforms.RandomEqualize(p=0.48), transforms.RandomResizedCrop(size=32, scale=(0.63, 0.97), ratio=(1.09, 1.59)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])