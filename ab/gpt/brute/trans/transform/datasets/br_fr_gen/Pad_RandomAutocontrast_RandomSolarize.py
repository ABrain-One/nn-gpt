import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(118, 192, 105), padding_mode='symmetric'),
    transforms.RandomAutocontrast(p=0.88),
    transforms.RandomSolarize(threshold=152, p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
