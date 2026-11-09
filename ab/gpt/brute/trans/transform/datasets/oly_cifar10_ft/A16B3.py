def transform(norm):
    return transforms.Compose([
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomAutocontrast(p=0.87),
        transforms.RandomGrayscale(p=0.23),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])