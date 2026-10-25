import torch
    import torchvision.transforms as transforms

    def transform(norm):
        return transforms.Compose([
            transforms.RandomRotation(degrees=(-15, 15), p=0.75),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1, p=0.25),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5), p=0.2),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(*norm)
        ])