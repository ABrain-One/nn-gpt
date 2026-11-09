<import>torchvision.transforms as transforms</import>
    <def>def transform(norm): return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.81),
    transforms.Pad(padding=0, fill=(173, 42, 189), padding_mode='edge'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.08, p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.52], std=[0.12], scale=True, mean_channels=[0.4], std_channels=[0.19]),
])</def>