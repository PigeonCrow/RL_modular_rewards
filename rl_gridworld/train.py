# %% imports
from agent import Agent
from env import RoomEnv
from reward import reward_function as reward_function

# from plots import plot_q_value_map, plot_rewards

# %% execution policy
if __name__ == "__main__":
    steps = 10000
    environment = RoomEnv(room_size=6)
    agent = Agent(
        env=environment,
        reward_function=reward_function,
        learning_rate=0.8,
    )
    agent_path = []
    environment.visualize_env()
    total_reward = 0
    reward_array = []
    for i in range(steps):
        s = environment.agent_position
        Qs = agent.choose_action()
        print(i,">>",Qs)
        action = agent.softmax_choice(Qs)
        # print(f"{i}: \t{action} \t{str(agent.V)}")
        snext = environment.step(action)
        # print(f"{environment.agent_position}")
        reward_array.append(agent.update_V(s, snext))
        total_reward += reward_array[-1]
        # print(f"{total_reward[-1]}")
        if total_reward > 0:
            environment.done = True
        if environment.agent_position == environment.reward_position:
            environment.done = True
        if environment.done:
            # print(f"FOUND REWARD AT STEP:{i}")
            # break
            pass
        agent_path.append((i, agent))
