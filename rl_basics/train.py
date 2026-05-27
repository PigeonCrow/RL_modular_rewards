# %%
import numpy as np
from agent import Agent, reward_function as rp
from env import RoomEnv



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
        print(action)
        # print(agent.V)
        snext = experiment.step(action)
        print(experiment.agent_position)
        reward = agent.update_V(s, snext)
        if reward > 0:
            experiment.done=True
        if experiment.agent_position == experiment.reward_position:
            experiment.done=True
        if experiment.done:
            print(f"EXIT AT EPOCH:{i}")
            break
    print(agent.V)
    print(experiment.agent_position)


# %%
main()
# %%
