from torch import nn


class Adapter(nn.Module):
    def __init__(self, size: int, reduction_factor: int = 16, out_size: int = None, init_std: float = .01):
        super().__init__()
        if out_size is None:
            out_size = size
        self.down_project = nn.Linear(size, size//reduction_factor)
        self.activation = nn.SiLU()
        self.up_project = nn.Linear(size//reduction_factor, out_size)

        self.down_project.weight.data.normal_(mean=0, std=init_std)
        self.down_project.bias.data.zero_()
        self.up_project.weight.data.normal_(mean=0, std=init_std)
        self.up_project.bias.data.zero_()

    def forward(self, x, residual=True):
        out = self.up_project(self.activation(self.down_project(x)))
        if residual:
            return x + out
        return out
