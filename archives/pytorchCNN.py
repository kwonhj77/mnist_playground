import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary

BATCH_SIZE = 128
EPOCHS = 15
NUM_CLASSES = 10

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), stride=1, padding=0)
        conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        maxPool = nn.MaxPool2d(kernel_size=(2,2))
        self.conv = nn.Sequential(
            conv1,
            nn.ReLU(),
            maxPool,
            conv2,
            nn.ReLU(),
            maxPool,
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)
        # self.fc = nn.Sequential(
        #     nn.Linear(1600, NUM_CLASSES),
        #     nn.Softmax()
        # )
        self.fc = nn.Linear(1600, NUM_CLASSES)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)

def train(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, device: torch.device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    # Load MNIST dataset
    train_dataset = datasets.MNIST(root='datasets\mnist', train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root='datasets\mnist', train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for x, y in train_loader:
        print(f"Train: Shape of image [N, C, H, W]: {x.shape}")
        print(f"Train: Shape of label: {y.shape} {y.dtype}")
        break
    for x, y in test_loader:
        print(f"Test: Shape of image [N, C, H, W]: {x.shape}")
        print(f"Test: Shape of label: {y.shape} {y.dtype}")
        break

    # Check if CUDA is available.
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU is available')
    else:
        device = torch.device('cpu')
        print('GPU not available, using CPU')
        print(f"Using {device} device")

    # Initialize model
    model = CNN().to(device)

    # Summary
    summary(model, input_size=(1, 28, 28))

    # Loss Fn and Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07)

    # Train and test
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer, device)
        test(test_loader, model, loss_fn, device)
    print("Done!")


if __name__ == '__main__':
    main()
    print("Exiting...")