import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.88, contrast=1.03, saturation=0.84, hue=0.07),
    transforms.RandomGrayscale(p=0.84),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.43),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
