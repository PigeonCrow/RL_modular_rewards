# %%
import numpy as np
from copy import deepcopy


# %%
class agent:
    def __init__(self, gamma, reward_function, env, learning_rate, beta):
        self.gamma = gamma
        self.reward_function = reward_function
        self.beta = beta
        self.lr = learning_rate
        self.env = env
        self.state = env.__init__()
        self.V = np.zeros((env.room_size, env.room_size))
        self.action_space = env.action_space()

    def update_V(self, s, snext):
        r = self.reward_function(self.env, s)
        V = self.V
        V[s[0]][s[1]] = V[s[0]][s[1]] + self.lr * (
            r + self.gamma * self.V[snext[0]][snext[1]] - V[s[0]][s[1]]
        )
        self.V = V
        return r

    def softmax_choice(self, Qs):
        ps = np.exp(self.beta * Qs) / np.sum(np.exp(self.beta * Qs))
        a = np.random.choice(len(ps), p=ps)
        return a

    def choose_action(self):
        Qs = np.zeros(len(self.action_space))
        env = self.env
        for a in self.action_space:
            s_hat = env.step(a)
            Qs[a] = self.V[s_hat[0]][s_hat[1]]
        a = self.softmax_choice(Qs)
        return a


# %%
def reward_function(
    env, position
):  # Return 1 if agent is at goal position or 0 otherwiser
    if env.agent_position == position:
        return 1
    else:
        return 0

# %%
