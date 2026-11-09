import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(224, 98, 11), padding_mode='constant'),
    transforms.RandomPerspective(distortion_scale=0.17, p=0.5),
    transforms.RandomAffine(degrees=17, translate=(0.05, 0.05), scale=(0.89, 1.67), shear=(1.3, 6.52)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
