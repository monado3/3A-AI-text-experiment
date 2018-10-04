import chainer
import chainerrl
from chainer import functions as F, links as L


class DQNModel(chainer.Chain, chainerrl.q_function.StateQFunction):
    """1次元ベクトルからQ値を返すQ関数モデル。

    Args:
        in_size: 入力される次元数
        out_size: 出力される次元数 (actionの数)
        gpu_id (int): GPU 番号 (GPUを使わない場合は None にする)
    """

    def __init__(self, in_size, out_size, gpu_id):
        unit_sizes = [in_size, 50, 50]
        super(DQNModel, self).__init__(
            l_1=L.Linear(unit_sizes[0], unit_sizes[1]),
            l_2=L.Linear(unit_sizes[1], unit_sizes[2]),
            l_out=L.Linear(unit_sizes[-1], out_size),
        )

    def __call__(self, x):
        h = x
        h = self.l_1(h)
        h = F.tanh(h)
        h = self.l_2(h)
        h = F.tanh(h)
        h = self.l_out(h)
        return chainerrl.action_value.DiscreteActionValue(h)
