# %%
import numpy as np
# from millidge_agent import Agent, reward_function as rp
from agent import Agent_millidge as Agent, reward_function as rp
from millidge_env import RoomEnv
from plots import plot_q_value_map, plot_q_value_maps
# %%
def main():
    steps = 10000
    n_agents = 6
    experiments = [RoomEnv() for x in range(0, n_agents)]
    # re = rp
    agents = [Agent(reward_function=rp,env=experiment,learning_rate=0.8) for experiment in experiments]
    for i in range(steps):
        s = [experiment.agent_position for experiment in experiments]
        Qs = [agent.choose_action() for agent in agents]
        # print(i)
        # print(action)
        # print(agent.V)
        # print(Qs)
        meta_q = sum(Qs)
        meta_action = agents[0].softmax_choice(meta_q)
        snext = [experiment.step(meta_action) for experiment in experiments]
        # print(experiment.agent_position)
        rewards = [agent.update_V(s[0], snext[0]) for agent in agents]
        # meta_V =  np.sum([agent.V for agent in agents])
        if len(rewards) > 0 and all(x >= 1 for x in rewards):
            experiments[0].done=True
            pass
        if experiments[0].agent_position == experiments[0].reward_position:
            experiments[0].done=True
            pass
        if experiments[0].done:
            print(f"EXIT AT STEP:{i}")
            break
    # print(agent.V)
    # print(experiment.agent_position)
    plot_q_value_map(experiments[0], agents[0].V)
    plot_q_value_maps(experiments, agents)

# %%
if __name__ == "__main__":
    main()
# %%
