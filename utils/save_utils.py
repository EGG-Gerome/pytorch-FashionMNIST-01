import os
import time
import torch

def save_model(model, metric, epoch, mode="acc", folder="checkpoints"):
    os.makedirs(folder, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"model_{mode}_{metric:.4f}_epoch{epoch}_{timestamp}.pth"
    filepath = os.path.join(folder, filename)
    torch.save(model.state_dict(), filepath)
    return filepath
