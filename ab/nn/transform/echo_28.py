import torchvision.transforms as transforms


def transform(_):
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()])
