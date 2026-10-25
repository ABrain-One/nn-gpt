def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.12, p=0.52),
    transforms.Pad(padding=1, fill=(164, 235, 196), padding_mode='edge'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])