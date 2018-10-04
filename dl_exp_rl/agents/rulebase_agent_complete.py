import chainerrl.agent
import gym


class RulebaseAgent(chainerrl.agent.Agent):
    """Rulebase agent

    ルールベース (人間が決め打ちした方策) で行動するagent
    穴埋めコードの解答編です

    Args:
        env (gym.Env): 環境 (観測・行動の空間を知るために使う)
        gpu_id (int): GPU 番号 (GPUを使わない場合は None にする)
    """

    def __init__(self, env, gpu_id):
        self.observation_num = {key: val.n for key, val in env.observation_space.spaces.items()}
        self.action_num = env.action_space.n

    def act_and_train(self, obs, reward):
        self.train(obs, reward)
        action = self.act(obs)
        return action

    def stop_episode_and_train(self, obs, reward, done=False):
        self.train(obs, reward)
        self.stop_episode()

    def act(self, obs):
        if obs['y'] != 2:
            action = 1
        else:
            action = 0
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
