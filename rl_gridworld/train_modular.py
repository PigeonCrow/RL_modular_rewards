# %% imports
import numpy as np
from agent import Agent
from env import RoomEnv
from reward import reward_function as reward_function

# from plots import plot_q_value_map, plot_q_value_maps, plot_rewards

# %%
if __name__ == "__main__":
    steps = 500
    n_agents = 3
    experiments = [RoomEnv() for x in range(0, n_agents)]
    agents = [
        Agent(
            env=experiment,
            reward_function=reward_function,
            learning_rate=0.8,
            gamma=0.3
        )
        for experiment in experiments
    ]
    total_rewards = np.zeros(n_agents)
    rewards_array = []
    for i in range(steps):
        s = [experiment.agent_position for experiment in experiments]
        Qs = [agent.choose_action() for agent in agents]
        meta_Q = np.sum(Qs, axis=1)
        action = agents[0].softmax_choice(meta_Q)
        # print(f"{i}: \t{action}")
        snext = [experiment.step(action) for experiment in experiments]
        rewards_array.append([agent.update_V(s[0], snext[0]) for agent in agents])
        # print(">>",i,">>",rewards_array[-1], ">>>", total_rewards)
        total_rewards = total_rewards + rewards_array[-1]
        for j, x in enumerate(total_rewards):
            if x > 0:
                experiments[j].done = True

        for experiment in experiments:
            if experiment.agent_position == experiment.reward_position:
                experiment.done = True

        if all([experiment.done for experiment in experiments]):
            # print(f"FOUND REWARD AT STEP:{i}")
            # break
            pass
