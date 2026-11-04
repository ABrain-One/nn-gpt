def transform(norm):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.34),
        transforms.CenterCrop(size=27),
        transforms.Pad(padding=2, fill=(200, 48, 184), padding_mode='value'),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])