import os

# 定义你的 raw 文件夹路径
raw_path = r"D:\develop\LeNet-5\data\MNIST\raw"

print(f"正在检查目录: {raw_path}")
print("-" * 30)

# 列出该目录下所有真实存在的文件
files = os.listdir(raw_path)
for f in files:
    print(f"发现文件: {f}")

# 检查 PyTorch 需要的特定文件是否存在
target_file = "train-images-idx3-ubyte.gz"
full_target = os.path.join(raw_path, target_file)

print("-" * 30)
if os.path.exists(full_target):
    print(f"成功找到目标文件: {target_file}")
else:
    print(f"未找到目标文件: {target_file}")
