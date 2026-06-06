import numpy as np


def reward_function(env):  # Return 1 if agent is at goal position or 0 otherwiser
    if env.agent_position == env.reward_position:
        return 1
    else:
        return -0.01

def dulberg_reward(env, h_star, h, n, m):
    if env.agent_position == env.reward_position:
        h = 1
    else:
        h = h - 0.01
    return np.abs(h_star - h) ** (n) ** (1 / m)
