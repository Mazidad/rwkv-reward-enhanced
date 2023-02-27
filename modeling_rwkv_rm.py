import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class RWKV_RM(nn.Module):
    def __init__(self, rwkv, base_model_