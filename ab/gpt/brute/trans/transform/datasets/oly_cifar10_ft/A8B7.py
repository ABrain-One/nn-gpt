def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=0, fill=(118, 234, 105), padding_mode='reflect'),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.86, 1.16)),
        transforms.RandomCrop(size=31),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])