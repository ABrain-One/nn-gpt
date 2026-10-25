import torch
import torchvision.transforms as transforms
from torchvision import transforms as T

def transform(norm):
    return T.Compose([T.RandomHorizontalFlip(p=0.4), T.RandomRotation(5), T.CenterCrop(28), T.RandomAutocontrast(p=0.5), T.Resize((64, 64)), T.ToTensor(), T.Normalize(**norm)])