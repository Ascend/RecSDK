import torch


def truncated_normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """
    使用截断正态分布初始化张量。

    该函数实现了截断正态分布(Truncated Normal Distribution), 用于初始化神经网络的权重。
    它通过生成一个均值为0, 标准差为1的标准正态分布, 然后将其截断到(-2, 2)区间内, 
    再将其缩放到指定的均值和标准差。

    参数:
    - x: 要初始化的张量。
    - mean: 初始化的均值。
    - std: 初始化的标准差。

    返回:
    - 初始化后的张量。
    """
    with torch.no_grad():
        size = x.shape
        tmp = x.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        x.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        x.data.mul_(std).add_(mean)
        return x