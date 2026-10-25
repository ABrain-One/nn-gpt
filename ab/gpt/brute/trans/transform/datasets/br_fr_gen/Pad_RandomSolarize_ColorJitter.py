import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(75, 146, 120), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=1, p=0.62),
    transforms.ColorJitter(brightness=1.16, contrast=0.85, saturation=0.95, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
