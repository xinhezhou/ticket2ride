import numpy as np
import torch
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from evaluation.evaluation import evaluate_net
from cnn_utils import load_net

def get_weights(model):
    weights  = {}
    state_dict = model.state_dict()
    for param in state_dict:
        weights[param] = torch.from_numpy(np.random.randn(*state_dict[param].size()))
    return weights


def update(weights, sigma, jitters):
    new_weights = {}
    for param in weights:
        jitter = torch.from_numpy(np.random.randn(*weights[param].size()))
        jitters[param].append(jitter)
        new_weights[param] = weights[param] + sigma * jitter
    return new_weights


def fitness(w, target_net, player_classes, opponent_net, destination_cards, first):
    target_net.load_state_dict(w)
    target_nets = [target_net, load_net(opponent_net)]
    if not first:
        target_nets = target_nets[:]
    winners, _ = evaluate_net(target_nets, player_classes, destination_cards, None, counter=100, display=False)
    return sum(winners) * first
    

def optimize_model(target_net, fitnesses, player_classes, opponent_net, destination_cards, first):
    npop = 1     # population size
    num_episodes = 1
    sigma = 0.1    # noise standard deviation
    alpha = 0.001  # learning rate
    w = get_weights(target_net)
    for i in range(num_episodes):
        R = np.zeros(npop)
        jitters = {}
        for param in w:
            jitters[param] = []
        for j in range(npop):
            w_try = update(w, sigma, jitters)
            R[j] = fitness(w_try,  target_net, player_classes, opponent_net, destination_cards, first)
        if np.sum(R) != 0:
            fitnesses.append(np.sum(R))
            A = (R - np.mean(R)) / np.std(R)
            for param in w:
                N = torch.stack(jitters[param])
                w[param] = w[param] + alpha/(npop*sigma) * np.dot(N.T, A)



def train_es(target_net, memory, checkpoint_file, BATCH_SIZE=128, GAMMA=0.9, TARGET_UPDATE=10, round=1000):

    target_net.eval()
    fitnesses = []

    for _ in range(round):
        optimize_model(target_net, memory, fitnesses, BATCH_SIZE, GAMMA)
    torch.save({
                'state_dict': target_net.state_dict(),
            }, checkpoint_file)
    plt.plot(range(len(fitnesses)), fitnesses)
    plt.show()




