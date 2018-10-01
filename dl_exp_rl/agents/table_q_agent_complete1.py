from collections import OrderedDict

import chainerrl.agent
import gym
import numpy as np


class TableQAgent(chainerrl.agent.Agent):
    """Table-Q agent

    ハッシュテーブルを用いてQ学習をするagent
    穴埋めコードの解答編1 です

    Args:
        env (gym.Env): 環境 (観測・行動の空間を知るために使う)
        gpu_id (int): GPU 番号 (GPUを使わない場合は None にする)
    """

    def __init__(self, env, gpu_id):
        self.observation_num = {key: val.n for key, val in env.observation_space.spaces.items()}
        self.action_num = env.action_space.n
        # 前回の観測・行動
        self.last_obs = None
        self.last_action = None
        # Q値
        self.q_table = {}
        # 学習率
        self.learning_rate = 0.1
        # 割引率
        self.discount_factor = 0.95
        # epsilon (ランダムに動く確率)
        self.exploration_prob = 0.3

    def act_and_train(self, obs, reward):
        self.train(obs, reward)
        return self.select_action(obs)

    def act(self, obs):
        return self.select_action(obs)

    def stop_episode_and_train(self, obs, reward, done=False):
        self.train(obs, reward)
        self.stop_episode()

    def stop_episode(self):
        self.last_obs = None
        self.last_action = None

    def save(self, dirname):
        pass

    def load(self, dirname):
        pass

    def get_statistics(self):
        return []

    def train(self, obs, reward):
        if self.last_obs is not None:
            assert (self.last_action is not None)
            last_obs_key, obs_key = [self.observation_to_key(o) for o in [self.last_obs, obs]]
            # 見たことないようなら辞書に追加
            if last_obs_key not in self.q_table:
                self.q_table[last_obs_key] = [0.0 for act in range(self.action_num)]

            # Q値の target を r + \gamma * max_a' Q(s', a') で求める
            if obs_key in self.q_table:
                max_q = max([self.q_table[obs_key][act] for act in range(self.action_num)])
            else:
                max_q = 0.0

            # Q学習をする
            self.q_table[last_obs_key][self.last_action] = (1 - self.learning_rate) * self.q_table[last_obs_key][
                self.last_action] + self.learning_rate * (reward + self.discount_factor * max_q)

        # 観測を保存
        self.last_obs = obs

    def select_action(self, obs):
        obs_key = self.observation_to_key(obs)
        if obs_key in self.q_table:
            # 観測から行動を決める
            action = self.epsilon_greedy(obs_key)
        else:
            # Q値がまだ定まっていないのでランダムに動く
            action = np.random.randint(self.action_num)
        self.last_action = action
        return action

    def observation_to_key(self, obs):
        return tuple(obs.values())

    def epsilon_greedy(self, obs_key):
        # 次の行動を epsilon-greedy( max_a Q(s, a) ) で決める
        # exploration (探索)
        random_action = np.random.randint(self.action_num)
        # exploitation (活用)
        max_q_action = np.argmax(self.q_table[obs_key])
        # どっちか選択
        action = np.random.choice([random_action, max_q_action],
                                  p=[self.exploration_prob, 1.0 - self.exploration_prob])
        return action

    def q_table_to_str(self):
        """Q table をいい感じに複数行の文字列にして返す"""

        def get_q(y, x, a):
            """obs=(y, x), action=a におけるQ値を返す"""
            obs_key = self.observation_to_key(OrderedDict(sorted([['y', y], ['x', x]])))
            if obs_key in self.q_table:
                return self.q_table[obs_key][a]
            else:
                return 0.0

        def get_format(j, i, y, x):
            """P = (y, x) における出力をうまいこと整形する
            (j, i) が
            (-1, -1) (-1,  0) (-1,  1)
            ( 0, -1) (0,0)(P) ( 0,  1)
            ( 1, -1) ( 1,  0) ( 1,  1)
            の位置に対応している
            """
            form_q = ' {0:04.2f} '
            form_space = ' ' * 6
            form_center = '({0:1d}, {1:1d})'
            d = abs(i) + abs(j)
            if d == 0:
                return form_center.format(y, x)
            elif d == 1:
                return form_q.format(get_q(y, x, [3, 1][(j + 1) // 2] if i == 0 else [2, 0][(i + 1) // 2]))
            else:
                return form_space

        return_str = ''
        for y in range(self.observation_num['y']):
            for tate in [-1, 0, 1]:  # [上段, 中段, 下段]
                for x in range(self.observation_num['x']):
                    return_str += ''.join([get_format(tate, yoko, y, x) for yoko in [-1, 0, 1]])
                return_str += '\n'
            return_str += '\n'
        return return_str
