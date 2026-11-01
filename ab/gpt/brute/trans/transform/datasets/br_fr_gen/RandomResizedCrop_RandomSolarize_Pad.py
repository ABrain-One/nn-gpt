import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.94), ratio=(1.05, 1.96)),
    transforms.RandomSolarize(threshold=218, p=0.51),
    transforms.Pad(padding=0, fill=(242, 105, 205), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
