import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.89, contrast=1.1, saturation=1.08, hue=0.07),
    transforms.Pad(padding=5, fill=(26, 167, 32), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
