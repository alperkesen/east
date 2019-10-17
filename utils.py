import torch
import torchvision

def loadpath(path):
    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=2,
        shuffle=True)

    return dataset, train_loader


def load_train():
    trainpath = "data/train"

    return loadpath(trainpath)


def load_test():
    testpath = "data/test"

    return loadpath(testpath)
