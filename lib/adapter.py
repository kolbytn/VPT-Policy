from torch import nn
import torch as th


class BaseAdapter(nn.Module):
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


class Adapter(nn.Module):
    def __init__(self, size: int, n_tasks: int = 1, reduction_factor: int = 16, out_size: int = None, init_std: float = .01):
        super().__init__()
        assert n_tasks >= 1
        self.task_adapters = nn.ModuleList([
            BaseAdapter(size, reduction_factor=reduction_factor, out_size=out_size, init_std=init_std)
            for _ in range(n_tasks)
        ])

    def forward(self, x, task_id=None, residual=True):
        if task_id is None:
            task_id = [0] * x.shape[0]
        assert len(task_id) == x.shape[0]
        out = []
        for i in range(len(task_id)):
            if task_id[i] == -1:
                out.append(x[i:i+1])
            else:
                assert task_id[i] >= 0 and task_id[i] < len(self.task_adapters), \
                    "Found out of range task id {}. Expected range [0, {}).".format(task_id, len(self.task_adapters))
                out.append(self.task_adapters[task_id[i]](x[i:i+1], residual=residual))
        return th.cat(out, dim=0)
