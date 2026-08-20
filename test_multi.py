import torch
from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from lenet5 import LeNet5

def test_model():
    # 1. 数据预处理（必须与训练时完全一致！）
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 保持与训练集相同的归一化
    ])

    # 2. 加载 MNIST 测试集
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,  # 设置为 False 表示加载测试集
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3. 初始化模型并加载权重
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    model.load_state_dict(torch.load('mnist_lenet5.pth', map_location=device))
    
    # 4. 将模型切换为评估模式（非常重要！）
    # 这会关闭 Dropout 和 BatchNorm 的训练行为，确保测试结果稳定
    model.eval()

    # 5. 开始测试（在测试阶段不需要计算梯度，以节省内存并加速）
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            
            # 获取预测概率最大的类别索引
            _, predicted = torch.max(output.data, 1)
            
            total += label.size(0)
            correct += (predicted == label).sum().item()

    # 6. 计算并打印准确率
    accuracy = 100 * correct / total
    print(f'模型在 MNIST 测试集上的精确度: {accuracy:.2f}%')

if __name__ == '__main__':
    test_model()
