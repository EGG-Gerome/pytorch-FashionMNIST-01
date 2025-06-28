from utils.dataLoader import train_dataloader,test_dataloader
from utils.bulid import build_optimizer, build_loss
from config import learning_rate, epochs, device
from utils.save_utils import save_model
from model.model import NeuralNetwork
from core.train_fn import train_fn
from core.test_fn import test_fn
from logger import Logger

best_loss = float('inf')
best_acc = 0.0
best_epoch_for_loss = 0
best_epoch_for_acc = 0

# 实例化模型、日志器、损失函数、优化器
model = NeuralNetwork().to(device)
logger = Logger()
loss_fn = build_loss("CrossEntropy")
optimizer = build_optimizer(model, optimizer_type="Adam", lr=learning_rate)

# 设置设备
logger.log(f"Using {device} device")

for epoch in range(epochs):
    logger.log_file.write(f"-------------------------------------------------------------------\n")
    print(f"-------------------------------------------------------------------")
    logger.log(f"Epoch {epoch+1}")
    train_fn(train_dataloader, model, loss_fn, optimizer, logger)

    test_loss, test_acc = test_fn(test_dataloader, model, loss_fn, logger)

    if test_loss < best_loss:
        best_loss = test_loss
        best_epoch_for_loss = epoch + 1

    if test_acc > best_acc:
        best_acc = test_acc
        best_epoch_for_acc = epoch + 1
    logger.log(f"Current best loss: {best_loss:.4f} at epoch {best_epoch_for_loss}")
    logger.log(f"Current best accuracy: {best_acc:.2f}% at epoch {best_epoch_for_acc}\n")

best_loss_filepath = save_model(model, metric=best_loss, epoch=best_epoch_for_loss, mode="loss")
best_acc_filepath = save_model(model, metric=best_acc, epoch=best_epoch_for_acc, mode="acc")
logger.log_file.write(f"-------------------------------------------------------------------\n")
print(f"-------------------------------------------------------------------")
logger.log_file.write(f"Best loss model saved to: {best_loss_filepath}\n")
print(f"Best loss model saved to: {best_loss_filepath}")
logger.log_file.write(f"Best acc model saved to: {best_acc_filepath}\n")
print(f"Best acc model saved to: {best_acc_filepath}")
logger.log("Done!")

logger.close()