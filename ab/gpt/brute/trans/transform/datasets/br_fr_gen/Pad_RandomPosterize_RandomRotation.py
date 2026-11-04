import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(71, 252, 228), padding_mode='reflect'),
    transforms.RandomPosterize(bits=4, p=0.18),
    transforms.RandomRotation(degrees=10),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
