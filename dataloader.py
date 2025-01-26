from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_mnist_data(batch_size):
    train_dataset = datasets.MNIST(root='datasets\mnist', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='datasets\mnist', train=False, download=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for x, y in train_loader:
        print(f"Train: Shape of image [N, C, H, W]: {x.shape}")
        print(f"Train: Shape of label: {y.shape} {y.dtype}")
        break
    for x, y in test_loader:
        print(f"Test: Shape of image [N, C, H, W]: {x.shape}")
        print(f"Test: Shape of label: {y.shape} {y.dtype}")
        break

    return train_loader, test_loader