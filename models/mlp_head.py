import torch.nn as nn
import torch

class MLPHead(nn.Module):
    """
    
    Cabeza de clasificacion estandar

    """

    def __init__(self,mlp_dims: list, dropout: float, act_function):
        super().__init__()
        self.dims = list(mlp_dims)
        self.dropout = dropout


        assert len(self.dims) >= 2

        layers = []

        for i in range(1,len(self.dims)):
            layers.append(nn.Linear(self.dims[i-1], self.dims[i]))
            if i < len(self.dims) - 1:
                layers.append(act_function())

                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
                               
        self.mlp_head = nn.Sequential(*layers)

    def forward(self,x):
        return self.mlp_head(x)

ACTIVATION_FUNCTION_REGISTRY = {
    'relu' : nn.ReLU,
    'gelu' : nn.GELU
}

def create_mlp_head(mlp_dims: list, dropout: float, act_name: str = 'relu'):
    if act_name not in ACTIVATION_FUNCTION_REGISTRY:
        raise ValueError(
            f"Activación '{act_name}' no soportada. "
            f"Disponibles: {list(ACTIVATION_FUNCTION_REGISTRY)}"
        )
    model = MLPHead(mlp_dims,dropout, act_function= ACTIVATION_FUNCTION_REGISTRY[act_name])
    return model
    

