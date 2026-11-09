import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.95), ratio=(0.94, 2.89)),
    transforms.Pad(padding=1, fill=(187, 181, 75), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=227, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
