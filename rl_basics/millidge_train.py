# %%
import numpy as np
from millidge_agent import Agent, reward_function as rp
from millidge_env import RoomEnv
from plots import plot_q_value_map, plot_q_value_maps
# %%
def main():
    epochs = 10000
    experiment1 = RoomEnv()
    experiment2 = RoomEnv()
    experiment3 = RoomEnv()
    # re = rp
    agent1 = Agent(
        reward_function=rp,
        env=experiment1,
        learning_rate=0.8,
    )
    agent2 = Agent(
        reward_function=rp,
        env=experiment2,
        learning_rate=0.8,
    )
    agent3 = Agent(
        reward_function=rp,
        env=experiment3,
        learning_rate=0.8,
    )
    for i in range(epochs):
        s = experiment1.agent_position # Common for all experiments
        action1, Q1 = agent1.choose_action()
        action2, Q2 = agent2.choose_action()
        action3, Q3 = agent3.choose_action()
        # print(i)
        # print(action)
        # print(agent.V)
        meta_q = Q1+Q2+Q3
        meta_action = agent1.softmax_choice(meta_q)
        snext1 = experiment1.step(meta_action)
        snext2 = experiment2.step(meta_action)
        snext3 = experiment3.step(meta_action)
        # print(experiment.agent_position)
        reward1 = agent1.update_V(s, snext1)
        reward2 = agent2.update_V(s, snext2)
        reward3 = agent3.update_V(s, snext3)
        # print(reward)
        if reward1 > 0:
            experiment1.done=True
            pass
        if experiment1.agent_position == experiment1.reward_position:
            experiment1.done=True
            pass
        if experiment1.done:
            print(f"EXIT AT EPOCH:{i}")
            break
        # plot_q_value_map(experiment1, agent1)
        # plot_q_value_map(experiment2, agent2)
        # plot_q_value_map(experiment3, agent3)
    # print(agent.V)
    # print(experiment.agent_position)
    plot_q_value_map(experiment1, agent1.V)
    plot_q_value_maps([experiment1, experiment2, experiment3], [agent1, agent2, agent3], meta_value=False)

# %%
if __name__ == "__main__":
    main()
# %%
