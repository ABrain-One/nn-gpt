import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.84, contrast=1.04, saturation=0.83, hue=0.04),
    transforms.Pad(padding=2, fill=(109, 93, 185), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
