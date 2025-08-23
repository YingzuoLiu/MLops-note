import gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class LoadBalanceEnv(gym.Env):
    """
    简单负载均衡 Gym 环境 (CPU-only demo)
    - action: 选择将任务分配给哪个 worker (0..n_workers-1)
    - observation: 各 worker 的当前离散负载 (integers)
    - reward: -延迟 (延迟 = 当前负载 + 噪声), 因此延迟越小奖励越大
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, n_workers=3, min_load=1, max_load=10):
        super().__init__()
        self.n_workers = n_workers
        self.min_load = min_load
        self.max_load = max_load

        # 动作空间：选择哪个 worker
        self.action_space = spaces.Discrete(n_workers)
        # 观测空间：每个 worker 的负载 (整数, 但 Gym 期待 float array)
        self.observation_space = spaces.Box(low=min_load, high=max_load,
                                            shape=(n_workers,), dtype=np.float32)
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        # 随机初始化各 worker 负载
        self.state = np.random.randint(self.min_load, 5, size=self.n_workers).astype(np.float32)
        self.step_count = 0
        return self.state

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        load = float(self.state[action])
        # 假设延迟 = load + [0,1) 噪声
        delay = load + np.random.rand()
        reward = -delay  # 延迟越小越好

        # 更新选中 worker 的负载（任务执行增加负载）
        self.state[action] = min(self.max_load, self.state[action] + np.random.randint(1, 3))
        # 其他 worker 负载下降（释放资源）
        for i in range(self.n_workers):
            if i != action:
                self.state[i] = max(self.min_load, self.state[i] - 1)

        self.step_count += 1
        done = False  # 本 demo 不终止 (你可按需添加条件)
        info = {"delay": delay}
        return self.state.copy(), reward, done, info

    def render(self, mode="human"):
        print("Loads:", self.state)

    def close(self):
        pass

# ---------------------------
# Q-learning (基于字典的状态表示)
# ---------------------------
def state_to_key(state, bins=10):
    # 将连续/整数状态转成 tuple_key，方便字典存储（这里直接取整数）
    # 若 state 为 float array，则取四舍五入
    ints = tuple(int(x) for x in state)
    return ints

def q_learning(env, episodes=200, steps_per_ep=20,
               alpha=0.1, gamma=0.95,
               epsilon=1.0, min_epsilon=0.05, decay=0.98):
    Q = {}  # dict: state_key -> action-value array
    ep_rewards = []
    eps_history = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for t in range(steps_per_ep):
            sk = state_to_key(state)
            if sk not in Q:
                Q[sk] = np.zeros(env.action_space.n)

            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[sk]))

            next_state, reward, done, info = env.step(action)
            nk = state_to_key(next_state)
            if nk not in Q:
                Q[nk] = np.zeros(env.action_space.n)

            # Q-learning 更新
            td_target = reward + gamma * np.max(Q[nk])
            td_error = td_target - Q[sk][action]
            Q[sk][action] += alpha * td_error

            state = next_state
            total_reward += reward

            if done:
                break

        # epsilon 衰减
        epsilon = max(min_epsilon, epsilon * decay)
        avg_reward = total_reward / steps_per_ep
        ep_rewards.append(avg_reward)
        eps_history.append(epsilon)

        print(f"Episode {ep+1:3d} | Avg Reward: {avg_reward: .3f} | Epsilon: {epsilon: .3f}")

    return Q, ep_rewards, eps_history

if __name__ == "__main__":
    env = LoadBalanceEnv(n_workers=3)
    episodes = 120
    steps_per_ep = 30

    Q, rewards, eps = q_learning(env,
                                 episodes=episodes,
                                 steps_per_ep=steps_per_ep,
                                 alpha=0.1, gamma=0.95,
                                 epsilon=1.0, min_epsilon=0.05, decay=0.96)

