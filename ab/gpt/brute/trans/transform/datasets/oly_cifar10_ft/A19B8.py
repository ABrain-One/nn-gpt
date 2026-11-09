def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(109, 233, 134), padding_mode='reflect'),
    transforms.RandomPosterize(bits=6, p=0.75),
    transforms.RandomGrayscale(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])