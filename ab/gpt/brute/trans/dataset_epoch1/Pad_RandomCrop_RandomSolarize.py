import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(181, 246, 197), padding_mode='reflect'),
    transforms.RandomCrop(size=30),
    transforms.RandomSolarize(threshold=197, p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
