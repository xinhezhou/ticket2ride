import numpy as np
import torch


def compute_dqn_input(graph, status, cards):
    array = np.concatenate([graph.flatten(), status.flatten(), cards])
    return torch.from_numpy(array).float()

def plot(rewards,losses, ax, batch_size):
    average_rewards = []
    average_losses = []
    num_episodes = len(rewards)

    # x = range(num_episodes)
    x = range(num_episodes // batch_size)
    for i in x:
        average_rewards.append(np.mean(rewards[i*batch_size: (i+1)*batch_size]))
        average_losses.append(np.mean(losses[i*batch_size: (i+1)*batch_size]))

    # x = range(num_episodes // BATCH_SIZE)
   
    ax[0].plot(x, average_rewards)
    ax[0].title.set_text("rewards")
    ax[1].plot(x, average_losses)
    ax[1].title.set_text("losses")