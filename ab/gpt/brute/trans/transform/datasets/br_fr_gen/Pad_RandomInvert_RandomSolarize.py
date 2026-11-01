import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(48, 66, 66), padding_mode='reflect'),
    transforms.RandomInvert(p=0.6),
    transforms.RandomSolarize(threshold=242, p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
