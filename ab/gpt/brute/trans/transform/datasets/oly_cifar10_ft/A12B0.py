<td>
def transform(norm):
    import torch
    import torchvision.transforms as transforms

    return transforms.Compose([
        transforms.RandomCrop(size=30),
        transforms.Pad(padding=2, fill=(109, 164, 195), padding_mode='symmetric'),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])
</td>