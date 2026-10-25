import torch
import torchvision.transforms as T

def transform(norm):
    return T.Compose([T.RandomApply([T.ConstantPad2d(4, (125, 122, 114))], p=1.0), T.RandomSolarize(threshold=167, p=0.13), T.RandomHorizontalFlip(p=0.14), T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), T.GaussianBlur(kernel_size=3, sigma=1.0), T.Resize((224, 224)), T.ToTensor(), T.Normalize(**norm)])