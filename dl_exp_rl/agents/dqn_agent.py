import chainer
import chainerrl
import gym
from dl_exp_rl.agents import models

from .utils import Utils


class DQNAgent(chainerrl.agents.DQN):
    """Deep Q-Network (ChainerRLの素晴らしい実装をラップしてるだけのクラス)

    このクラスを呼び出すと、DQNがよしなに初期化されて返る
    観測をいい感じにflattenする処理もここで作る

    Args:
        env (gym.Env): 環境 (観測・行動の空間を知るために使う)
        gpu_id (int): GPU 番号 (GPUを使わない場合は None にする)
    """

    def __init__(self, env, gpu_id):
        # 前処理関数を作る
        obs_space = env.observation_space
        flatten, obs_size = Utils.get_flatten_function_and_size(obs_space)
        n_actions = env.action_space.n

        # model (neural network) を作る
        q_func = models.DQNModel(obs_size, n_actions, gpu_id)
        if gpu_id is not None:
            chainer.cuda.get_device_from_id(gpu_id).use()
            q_func.to_gpu()

        # オプティマイザ (勾配降下法を賢くやってくれる)
        optimizer = chainer.optimizers.Adam(eps=1e-2)
        optimizer.setup(q_func)

        # 探索のしかた
        explorer = chainerrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)

        # replay buffer (過去の履歴をまとめておくことで時系列性をなくせる)
        replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 5)
        replay_start_size = 200

        # 割引率
        discount_factor = 0.95

        # chainerrl.agents.DQN の __init__() を呼び出す
        super().__init__(q_func, optimizer, replay_buffer, discount_factor, explorer,
                         gpu=None, replay_start_size=replay_start_size,
                         minibatch_size=32, update_interval=1,
                         target_update_interval=100, phi=flatten,
                         average_q_decay=0.95)
