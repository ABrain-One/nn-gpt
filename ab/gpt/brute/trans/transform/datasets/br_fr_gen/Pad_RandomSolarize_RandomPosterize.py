import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(94, 190, 193), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=14, p=0.43),
    transforms.RandomPosterize(bits=8, p=0.51),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
