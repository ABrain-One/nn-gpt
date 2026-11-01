import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(148, 102, 148), padding_mode='constant'),
    transforms.RandomSolarize(threshold=90, p=0.11),
    transforms.RandomCrop(size=24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
