import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(50, 111, 220), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.57),
    transforms.RandomAutocontrast(p=0.29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
