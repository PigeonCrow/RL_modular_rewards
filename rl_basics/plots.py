import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

# sns.set_theme('talk', font_scale=1.2)
# cmap = sns.color_palette('Blues', as_cmap=True)


def plot_rewards():
    pass


# def plot_q_value_map(env, agent):
#     value_map = policy_map = agent.V
#     arrow_map = {
#         0: "v",  # Down
#         1: ">",  # Right
#         2: "^",  # Up
#         3: "<",  # Left
#     }
#     fig, ax = plt.subplots(figsize=(7, 7))
#     cax = ax.imshow(value_map, cmap="winter", norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10))
#     m = np.nanmax(np.abs(value_map))
#     cax = ax.imshow(value_map, cmap="viridis", norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10, vmin=-m, vmax=m))
#     fig.colorbar(cax, ax=ax, label="Log of State Value (max Q-value)")
#     for r in range(env.room_size):
#         for c in range(env.room_size):
#             text_color = "white" if policy_map[r, c] in arrow_map.values() else "black"
#             ax.text(
#                 c,
#                 r,
#                 policy_map[r, c].round(4),
#                 ha="center",
#                 va="center",
#                 color=text_color,
#                 fontsize=12,
#                 weight="normal",
#             )
#     ax.set_title("Learned Policy and State Values", fontsize=16)
#     ax.set_xticks([])  # Hide x-axis ticks
#     ax.set_yticks([])  # Hide y-axis ticks
#     plt.tight_layout()
#     plt.show()

#     pass

# def plot_reward_function(env, rfun, title=""):
#     # env = RoomEnv()
#     room_size = env.room_size
#     mat = np.zeros((room_size,room_size ))
#     for i in range(room_size):
#         for j in range(room_size):
#             env.agent_position = [i,j]
#             r = rfun(env)
#             mat[i, j] = r

#     fig = plt.figure()
#     sns.heatmap(cmap=cmap, data=mat, vmin=-0.1, vmax=5)
#     plt.xticks([])
#     plt.yticks([])
#     if title != "":
#         plt.title(title)
#     # save_figure(sname)
#     plt.show()


def plot_q_value_map(env, agent_V):
    value_map = agent_V
    fig, ax = plt.subplots(figsize=(7, 7))

    m = np.nanmax(np.abs(value_map))
    cax = ax.imshow(
        value_map,
        cmap="viridis",
        norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10, vmin=-m, vmax=m),
    )

    fig.colorbar(cax, label="Log of State Value (max Q-value)")

    for r in range(env.room_size):
        for c in range(env.room_size):
            ax.text(
                c,
                r,
                f"{value_map[r, c]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(value_map[r, c]) > 0.5 * m else "black",
                fontsize=12,
            )

    ax.set_title("Learned Policy and State Values", fontsize=16)
    ax.grid(True, alpha=0.2)
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_step_wise_q_value(env, Qs):

    pass


def plot_q_value_map_ax(ax, env, agent_V):
    value_map = agent_V
    # print(value_map)
    m = np.nanmax(np.abs(value_map))
    cax = ax.imshow(
        value_map,
        cmap="viridis",
        norm=SymLogNorm(linthresh=1e-2, linscale=1, base=10, vmin=-m, vmax=m),
    )
    for r in range(env.room_size):
        for c in range(env.room_size):
            ax.text(
                c,
                r,
                f"{value_map[r, c]:.4f}",
                ha="center",
                va="center",
                color="white" if abs(value_map[r, c]) > 0.5 * m else "black",
                fontsize=8,
            )
    ax.grid(True, alpha=0.2)
    # ax.set_xticks([])
    # ax.set_yticks([])

    return cax


def plot_q_value_maps(
    envs,
    agents,
    ncols=3,
    figsize_per_plot=4,
    meta_value=True,
    suptitle="Learned Policies and State Values",
):
    if len(agents) not in (1, len(envs)):
        raise ValueError("agents must be length 1 or same length as envs")
    N = len(envs)
    if meta_value:
        N += 1
    ncols = max(1, int(ncols))
    nrows = (N + ncols - 1) // ncols
    # Use constrained_layout to avoid tight_layout warning with colorbars / suptitle
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_plot, nrows * figsize_per_plot),
        constrained_layout=True,
    )

    # normalize axes array
    axes = np.atleast_1d(axes).flatten()

    last_cax = None
    for i, env in enumerate(envs):
        ax = axes[i]
        ax.set_title(f"Agent-{i}")
        # agent = agents[0] if len(agents) == 1 else agents[i]
        agent = agents[i]
        last_cax = plot_q_value_map_ax(ax, env, agent.V)

    if meta_value:
        ax = axes[i + 1]
        ax.set_title("Meta Q-Values")
        meta_V = np.sum([agent.V for agent in agents], axis=0)
        # print("meta", meta_V)
        last_cax = plot_q_value_map_ax(ax, env, meta_V)

    for j in range(N, len(axes)):
        axes[j].axis("off")

    if last_cax is not None:
        fig.colorbar(
            last_cax,
            ax=axes[:N].tolist(),
            label="Log of State Value (max Q-value)",
            shrink=0.8,
        )

    fig.suptitle(suptitle, fontsize=16)
    plt.show()
