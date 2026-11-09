import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.13, contrast=1.08, saturation=0.96, hue=0.06),
    transforms.Pad(padding=2, fill=(124, 48, 127), padding_mode='symmetric'),
    transforms.CenterCrop(size=31),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
