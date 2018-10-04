# 演習解答欄

## 演習 1.1.2

作成している行
agent:`agent = agents.RandomAgent(env, gpu_id)`
env:`env = gym.make('EasyMaze-v0')`


呼び出されている変数・メソッド

agent:
* `act_and_train()`
* `act()`
* `get_statistics()`
* `stop_episode_and_train()`
* `stop_episode()`

env:
* `metadata`
* `render()`
* `step()`

## 演習 1.1.3

平均step数

- train:`46.08`
- test:`47.81`

## 演習 1.1.8

平均step数

- train: `9.5`
    - train first 10: `22.6`
    - train last 10: `7.8`
- test: `7.21`


## 演習 1.1.10

平均step数

train:
test:
