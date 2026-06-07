# %% imports
import numpy as np


# %% Agent class
class Agent:
    def __init__(
        self,
        env,
        reward_function,
        gamma=0.99,  # Discount factor for future rewards
        learning_rate=0.9,  # learning rate
        beta=0.8,  # action policy weight
    ):
        self.env = env
        self.reward_function = reward_function
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.beta = beta
        self.V = np.zeros(
            (self.env.room_size, self.env.room_size)
        )  # (self.env.room_size * self.env.room_size) Change for single array V
        self.action_space = self.env.action_space()
        self.reward = 0
        self.random_policy = False  # For not just randomly positioning agent

    def update_V(self, s, snext):
        r = self.reward_function(self.env)
        V = self.V
        V[s[0]][s[1]] = V[s[0]][s[1]] + self.learning_rate * (r + self.gamma * V[snext[0]][snext[1]] - V[s[0]][s[1]])
        self.V = V
        return r

    def softmax_choice(self, Qs):
        ps = np.exp(self.beta * Qs) / np.sum(np.exp(self.beta * Qs))
        # print("ps",ps)
        a = np.random.choice(len(ps), p=ps)
        return a

    def choose_action(self):
        if self.random_policy:
            a = np.random.choice(len(self.action_space))
            return a
        Qs = np.zeros(len(self.action_space))
        env = self.env
        for a in self.env.action_space():
            s_hat = env.step(a, simulated=True)
            Qs[a] = self.V[s_hat[0]][s_hat[1]]
        a = self.softmax_choice(Qs)
        env.step(a, simulated=True)
        return Qs


# %% linear agent
class Agent_linear:  # alternative for linear V array
    def __init__(
        self,
        env,
        reward_function,
        gamma=0.99,  # Discount factor for future rewards
        learning_rate=0.9,  # learning rate
        beta=0.8,  # action policy weight
    ):
        self.env = env
        self.reward_function = reward_function
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.beta = beta
        self.V = np.zeros(self.env.room_size * self.env.room_size)
        self.action_space = self.env.action_space()
        self.reward = 0
        self.random_policy = False  # For not just randomly positioning agent

    def update_V(self, s, snext):  # alternative for linear V array
        r = self.reward_function(self.env)
        V = self.V
        s_idx = self.env.find_idx(s)
        snext_idx = self.env.find_idx(snext)
        V[s_idx] = V[s_idx] + self.learning_rate * (r + self.gamma * V[snext_idx] - V[s_idx])
        self.V = V
        return r

    def softmax_choice(self, Qs):
        ps = np.exp(self.beta * Qs) / np.sum(np.exp(self.beta * Qs))
        a = np.random.choice(len(ps), p=ps)
        return a

    def choose_action(self):
        if self.random_policy:
            a = np.random.choice(len(self.action_space))
            return a
        Qs = np.zeros(len(self.action_space))
        env = self.env
        for a in self.env.action_space():
            s_hat = env.step(a, simulated=True)
            s_idx = self.env.find_idx(s_hat)
            Qs[a] = self.V[s_idx]
        a = self.softmax_choice(Qs)
        env.step(a, simulated=True)
        return Qs


# %% Millidge Agent
# uses motivation in update V step
class Agent_millidge(Agent):
    def __init__(
        self,
        reward_function,
        env,
        gamma=0.99,  # Discount factor for future rewards
        learning_rate=0.9,  # learning rate
        beta=0.8,  # action policy weight
        motivation=0.5,
    ):
        super().__init__(
            reward_function,
            env,
            gamma=gamma,
            learning_rate=learning_rate,
            beta=beta,
        )
        self.motivation = motivation

    def update_V(self, s, snext):
        r = self.reward_function(self.env)
        V = self.V
        motivation = self.motivation
        V[s[0]][s[1]] = motivation * (
            V[s[0]][s[1]]
            + self.learning_rate * (r + self.gamma * self.V[snext[0]][snext[1]] - self.gamma**2 * V[s[0]][s[1]])
        )
        self.V = V
        return r


# %% Dulberg Agent
# uses an (h_star - h) for motivation from environment
class Agent_dulberg(Agent):
    def update_V(self, s, snext, h_star, h, n, m):
        r = self.reward_function(self.env, h_star, h, n, m)
        V = self.V
        V[s[0]][s[1]] = V[s[0]][s[1]] + self.learning_rate * (
            r + self.gamma * self.V[snext[0]][snext[1]] - self.gamma**2 * V[s[0]][s[1]]
        )
        self.V = V
        return r


# %%
if __name__ == "__main__":
    print("Agents?")

# %%
