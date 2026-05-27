import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


def plot_rewards():
    pass


def plot_q_value_map(env, agent):
    value_map = policy_map = agent.V
    arrow_map = {
        0: "v",  # Down
        1: ">",  # Right
        2: "^",  # Up
        3: "<",  # Left
    }
    fig, ax = plt.subplots(figsize=(7, 7))
    cax = ax.imshow(value_map, cmap="winter", norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10))
    m = np.nanmax(np.abs(value_map))
    cax = ax.imshow(value_map, cmap="managua", norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10, vmin=-m, vmax=m))
    fig.colorbar(cax, ax=ax, label="Log of State Value (max Q-value)")
    for r in range(env.room_size):
        for c in range(env.room_size):
            text_color = "white" if policy_map[r, c] in arrow_map.values() else "black"
            ax.text(
                c,
                r,
                policy_map[r, c].round(4),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                weight="normal",
            )
    ax.set_title("Learned Policy and State Values", fontsize=16)
    ax.set_xticks([])  # Hide x-axis ticks
    ax.set_yticks([])  # Hide y-axis ticks
    plt.show()

    pass


def plot_step_wise_q_value():
    pass
