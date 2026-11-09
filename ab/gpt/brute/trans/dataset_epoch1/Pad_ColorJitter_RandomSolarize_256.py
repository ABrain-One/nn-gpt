import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(231, 98, 54), padding_mode='symmetric'),
    transforms.ColorJitter(brightness=0.89, contrast=0.91, saturation=1.1, hue=0.02),
    transforms.RandomSolarize(threshold=104, p=0.73),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
