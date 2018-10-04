from collections import OrderedDict

import chainerrl.agent
import gym


class RulebaseAgent(chainerrl.agent.Agent):
    """Rulebase agent

    ルールベース (人間が決め打ちした方策) で行動するagent
    穴埋めコードです

    Args:
        env (gym.Env): 環境 (観測・行動の空間を知るために使う)
        gpu_id (int): GPU 番号 (GPUを使わない場合は None にする)
    """

    def __init__(self, env, gpu_id):
        self.observation_num = {key: val.n for key, val in env.observation_space.spaces.items()}
        self.action_num = env.action_space.n
        self.action = 0
        self.prev_y = None

    def act_and_train(self, obs, reward):
        self.train(obs, reward)
        action = self.act(obs)
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        self.train(obs, reward)
        self.stop_episode()

    def act(self, obs: OrderedDict) -> int:
        """
        action
         0:right 1:down 2:left 3:top

         :param obs:現在位置の座標 {'x':int, 'y':int}
         :return: どっちに進むかを表す整数
        """

        # ---穴埋め---
        # 観測を利用して、最短ステップでゴールできるようなactionを返せるようにせよ。
        # 例えば、
        # print(obs)
        # などとすることで、観測がどのように与えられるかを確認することができる。
        # また、
        # action = 0
        # として実行すれば、「action = 0」が迷路において
        # どのような行動に相当するかを見ることができる。
        # ------------
        # if  # here #  :
        #     action =  # here #
        # else:
        #     action =  # here #
        now_y = obs['y']
        if now_y == self.prev_y:
            action = 0
        else:
            action = 1
        self.prev_y = now_y
        return action

    def train(self, obs, reward):
        pass

    def stop_episode(self):
        pass

    def save(self, dirname):
        pass

    def load(self, dirname):
        pass

    def get_statistics(self):
        return []
