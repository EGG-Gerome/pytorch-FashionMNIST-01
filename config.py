import torch

learning_rate = 1e-3
batch_size = 64
epochs = 2

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
