# %%
import matplotlib.pyplot as plt
import numpy as np

# from plots import plot_reward_function
# from agent import reward_function


class RoomEnv:  # room with 1 agent, all square are 0 and agent is 1 #(BaseEnv):
    def __init__(
        self,
        room_size=6,
        start_position=[0, 0],
        reward_position=None,
    ):
        self.room_size = room_size  # Length of square
        self.start_point = start_position  # start position
        self.agent_position = self.start_point
        print(room_size)
        if reward_position is None:
            self.reward_position = [room_size - 1, room_size - 1]
            # self.reward_position = [np.random.randint(0,self.room_size), np.random.randint(0, self.room_size)]
        else:
            self.reward_position = reward_position
        print("reward is at ", self.reward_position)
        self.state = np.zeros((room_size, room_size))  # empty room (all squares are 0)
        self.state[start_position[0], start_position[1]] = (
            1  # position the agent at start_position[x,y]
        )
        print(reward_position)
        self.state[self.reward_position[0], self.reward_position[1]] = 3
        self.done = False  # status of task
        self.rewards = 0  # rewards from task

    def action_space(self):
        return (0, 1, 2, 3)

    def step(self, a=None, simulated=True):
        agent_position_x, agent_position_y = self.agent_position
        room_size = self.room_size
        if a == 0:  # Move up
            if not agent_position_y >= room_size - 1:  # Check if agent can move up
                agent_position_y += 1
        elif a == 1:  # Move right
            if not agent_position_x >= room_size - 1:  # Check if agent can move right
                agent_position_x += 1
        elif a == 2:  # Move Down
            if not agent_position_y <= 0:  # Check if agent can move down
                agent_position_y -= 1
        elif a == 3:  # Move Left
            if not agent_position_x <= 0:  # Check if agent can move left
                agent_position_x -= 1
        if not simulated:
            self.agent_position = [agent_position_x, agent_position_y]
        return [agent_position_x, agent_position_y]

    def terminate(self, pos):
        if self.state[pos[0], pos[1]] != 0:
            self.done = True
            return True

    def reset(self):
        self.rewards = 0  # reset rewards to 0
        self.state[self.start_point[0], self.start_point[1]] = (
            1  # set agent to start position
        )
        self.agent_position = self.start_point

    def visualize_env(self):
        # fig, ax = plt.subplots(figsize=(7, 7))
        print(self.state.shape)
        plt.grid(True, alpha=0.2)
        plt.imshow(self.state)
        plt.xticks(range(self.room_size))
        plt.yticks(range(self.room_size))
        plt.tight_layout()
        plt.show()


# %%
def main():
    print("RUNNING")
    room_env = RoomEnv(room_size=10)
    room_env.visualize_env()
    # plot_reward_function(room_env, reward_function  )


# %%
if __name__ == "__main__":
    main()
# %%
