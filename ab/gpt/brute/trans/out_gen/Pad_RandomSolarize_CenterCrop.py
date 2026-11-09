import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(211, 215, 169), padding_mode='reflect'),
    transforms.RandomSolarize(threshold=32, p=0.35),
    transforms.CenterCrop(size=24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
