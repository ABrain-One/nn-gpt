import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(51, 161, 165), padding_mode='constant'),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.97), ratio=(0.84, 2.66)),
    transforms.RandomSolarize(threshold=5, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
