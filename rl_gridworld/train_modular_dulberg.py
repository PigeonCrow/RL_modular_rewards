# %%

import numpy as np
from agent import Agent_dulberg as Agent
from env import RoomEnv
from plot import plot_q_value_map, plot_q_value_maps  # , plot_rewards
from reward import dulberg_reward as reward_function


# %%
def main():
    steps = 500
    n_agents =3
    experiments = [RoomEnv() for x in range(0, n_agents)]
    agents = [
        Agent(
            reward_function=reward_function,
            env=experiment,
            learning_rate=0.8,
            beta=0.1
        )
        for experiment in experiments
    ]
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
        meta_Q = np.sum(Qs, axis=0)
        print(i,">>",meta_Q)
        action = agents[0].softmax_choice(meta_Q)
        # print(f"{i}: \t{action}")
        snext = [experiment.step(action) for experiment in experiments]
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
    plot_q_value_maps(experiments, agents)
    # plot_rewards(np.vstack(reward_track).T[0])
    # plot_rewards(np.vstack(reward_track).T[1])
    # plot_rewards(np.vstack(reward_track).T[2])


# %%
if __name__ == "__main__":
    main()
# %%
