import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(151, 176, 0), padding_mode='symmetric'),
    transforms.RandomGrayscale(p=0.67),
    transforms.RandomHorizontalFlip(p=0.66),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
