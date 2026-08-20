import torch
from torchvision import transforms
from PIL import Image
from lenet5 import LeNet5

def predict_single_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    model.load_state_dict(torch.load('mnist_lenet5.pth', map_location=device))
    model.eval()

    img = Image.open(image_path)
    img = transform(img)
    
    img = torch.unsqueeze(img, dim=0).to(device) 

    with torch.no_grad():
        output = model(img)
        print(output.data)
        _, predicted = torch.max(output.data, 1)
        
    return predicted.item()


if __name__ == '__main__':
    for i in range(10):
        image_path = f'./test_images/{i}.png'  
        result = predict_single_image(image_path)
        print(f"该图片预测的数字是: {result},实际结果是：{i}")
