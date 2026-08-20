import torch
from torch import nn
from torch import optim

from lenet5 import LeNet5

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',       
        train=True,          
        download=True,       
        transform=transform
    )
    print("train_dataset length:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    print("train_loader length:", len(train_loader))

    model = LeNet5()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()


    for epoch in range(50):
        for batch_idx, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"轮次 {epoch + 1}/50 "
                      f"| 小批量 {batch_idx}/{len(train_loader)} "
                      f"| 损失: {loss.item():.4f}")
    torch.save(model.state_dict(), 'mnist_lenet5.pth')
