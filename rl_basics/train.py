# %%
from agent import Agent
from agent import reward_function as rp
from env import RoomEnv
from plots import plot_q_value_map


# %%
def main():
    steps = 10000
    experiment = RoomEnv(room_size=6)
    # re = rp
    agent = Agent(
        reward_function=rp,
        env=experiment,
        learning_rate=0.8,
    )
    step_agent = []
    experiment.visualize_env()
    for i in range(steps):
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
            experiment.done = True
            pass
        if experiment.agent_position == experiment.reward_position:
            experiment.done = True
            pass
        if experiment.done:
            print(f"EXIT AT STEP:{i}")
            break
        # plot_q_value_map(experiment, agent)
        step_agent.append((i, agent))

    # print(agent.V)
    # print(experiment.agent_position)
    plot_q_value_map(experiment, agent.V)


# %%
if __name__ == "__main__":
    main()
# %%
