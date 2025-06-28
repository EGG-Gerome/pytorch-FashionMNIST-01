from config import device
import torch

def test_fn(dataloader, model, loss_fn, logger):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_acc /= size


    logger.log(f"Test Performance: \n Accuracy: {(100 * test_acc):>0.1f}%, Avg loss: {test_loss:>8f}")

    return test_loss, test_acc
