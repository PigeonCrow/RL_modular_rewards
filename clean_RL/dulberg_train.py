# %%

# from millidge_agent import Agent, reward_function as rp
import numpy as np
from agent import Agent_dulberg as Agent
from agent import dulberg_reward as rp
from env import RoomEnv
from plots import plot_q_value_map, plot_q_value_map_ax, plot_q_value_maps, plot_rewards


# %%
def main():
    steps = 500
    n_agents = 3
    experiments = [RoomEnv() for x in range(0, n_agents)]
    # re = rp
    agents = [Agent(reward_function=rp, env=experiment, learning_rate=0.8) for experiment in experiments]
    h_star = np.ones(n_agents)  # [1, 1, 1, ....]
    h = np.random.rand(n_agents)  # [0.5, 0.1, 0.3....]
    h[0] = 1
    print(h_star, h)
    n = 4
    m = 2
    rewards = np.zeros(n_agents)
    reward_track = []
    for i in range(steps):
        s = [experiment.agent_position for experiment in experiments]
        Qs = [agent.choose_action() for agent in agents]
        # print(i)
        # print(action)
        # print(agent.V)
        # print(Qs)
        meta_q = np.sum(Qs, axis=1)
        meta_action = agents[0].softmax_choice(meta_q)
        # meta_V =  np.sum([agent.V for agent in agents])
        snext = [experiment.step(meta_action) for experiment in experiments]
        # print(experiment.agent_position)
        rewards = rewards + [agent.update_V(s[0], snext[0], h_star[i], h[i], n, m) for i, agent in enumerate(agents)]
        # print(rewards)
        reward_track.append(rewards)
        # meta_V =  np.sum([agent.V for agent in agents])
        if experiments[0].agent_position == experiments[0].reward_position:
            experiments[0].done = True
            pass
        if experiments[0].done:
            # print(f"EXIT AT STEP:{i}")
            # break
            pass
    # print(agent.V)
    # print(experiment.agent_position)
    plot_q_value_map(experiments[0], agents[0].V)
    plot_q_value_map_ax(ax=None,env=experiments[0], agent_V=agents[0].V)
    plot_q_value_maps(experiments, agents)
    plot_rewards(np.vstack(reward_track).T[0])
    # plot_rewards(np.vstack(reward_track).T[1])
    # plot_rewards(np.vstack(reward_track).T[2])


# %%
if __name__ == "__main__":
    main()
# %%
