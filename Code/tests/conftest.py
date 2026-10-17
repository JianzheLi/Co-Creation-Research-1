# Set the Windows MKL policy before any numerical libraries are imported.
import src
import torch
torch.set_num_threads(2)
