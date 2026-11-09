import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(198, 105, 105), padding_mode='reflect'),
    transforms.RandomPosterize(bits=4, p=0.53),
    transforms.RandomAffine(degrees=18, translate=(0.04, 0.18), scale=(0.84, 1.29), shear=(2.85, 9.04)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
