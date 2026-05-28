# %%
import numpy as np
from copy import deepcopy


# %%
class Agent:
    def __init__(
        self,
        reward_function,
        env,
        gamma=0.99,
        learning_rate=0.1,
        beta=1,
    ):
        self.gamma = gamma
        self.reward_function = reward_function
        self.beta = beta
        self.lr = learning_rate
        self.env = env
        self.V = np.zeros((env.room_size, env.room_size))
        self.action_space = env.action_space()

    def update_V(self, s, snext):
        r = self.reward_function(self.env)
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
        for a in self.env.action_space():
            s_hat = env.step(a, simulated=True)
            Qs[a] = self.V[s_hat[0]][s_hat[1]]
        a = self.softmax_choice(Qs)
        env.step(a, simulated=False)
        return a

#%%
class Agent_millidge(Agent):
    def choose_action(self):
        Qs = np.zeros(len(self.action_space))
        env = self.env
        for a in self.env.action_space():
            s_hat = env.step(a, simulated=True)
            Qs[a] = self.V[s_hat[0]][s_hat[1]]
        a = self.softmax_choice(Qs)
        env.step(a, simulated=False)
        return Qs

# %%
def reward_function(env):  # Return 1 if agent is at goal position or 0 otherwiser
    if env.agent_position == env.reward_position:
        return 1
    else:
        return -0.1

# %%
