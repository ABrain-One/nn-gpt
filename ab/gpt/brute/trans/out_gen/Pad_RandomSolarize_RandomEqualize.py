import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(4, 219, 68), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=142, p=0.35),
    transforms.RandomEqualize(p=0.83),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
