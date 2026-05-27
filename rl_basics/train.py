# %%
import numpy as np
from agent import Agent, reward_function as rp
from env import RoomEnv
from plots import plot_q_value_map
# %%
def main():
    epochs = 10000
    experiment = RoomEnv()
    # re = rp
    agent = Agent(
        reward_function=rp,
        env=experiment,
        learning_rate=0.8,
    )
    for i in range(epochs):
        s = experiment.agent_position
        action = agent.choose_action()
        # print(i)
        # print(action)
        # print(agent.V)
        snext = experiment.step(action)
        # print(experiment.agent_position)
        reward = agent.update_V(s, snext)
        # print(reward)
        if reward > 0:
            experiment.done=True
            pass
        if experiment.agent_position == experiment.reward_position:
            experiment.done=True
            pass
        if experiment.done:
            print(f"EXIT AT EPOCH:{i}")
            break
        # plot_q_value_map(experiment, agent)
        
    # print(agent.V)
    # print(experiment.agent_position)
    plot_q_value_map(experiment, agent)

# %%
if __name__ == "__main__":
    main()
# %%
