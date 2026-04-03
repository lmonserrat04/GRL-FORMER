import random
import numpy as np
import torch
import os

def set_seed(seed):
    """
    Establece la semilla aleatoria para reproducibilidad en:
    random, numpy, torch (CPU y GPU), y variables de entorno.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Para multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Hace que cuDNN sea determinista (más lento pero reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False