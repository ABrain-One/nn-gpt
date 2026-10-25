import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=0.96, saturation=0.98, hue=0.06),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomSolarize(threshold=244, p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
