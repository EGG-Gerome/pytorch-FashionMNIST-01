from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from config import batch_size

# 下载数据集，整理数据集， 创建Dataset 实例
# Downloading the train data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

# Downloading the test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data,batch_size=batch_size)

# 检查从 DataLoader 中取出的数据是否格式正确
# for X, y in test_dataloader:
#     print(f"Shape of X [N, C, H, W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break