import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from solitaire import play_game
sys.path.append("../")
from RLplayers.cnn_network import CNNSimple
from RLplayers.cnn_player import CNNPlayer
from RLplayers.rl_utils import load_net
from game import Game



def get_weights(model):
    weights  = {}
    state_dict = model.state_dict()
    for param in state_dict:
        weights[param] = torch.from_numpy(np.random.randn(*state_dict[param].size()))
    return weights


def update(weights, sigma, jitters):
    new_weights = {}
    for param in weights:
        jitter = torch.from_numpy(np.asarray(np.random.randn(*weights[param].size())))
        jitters[param].append(jitter)
        new_weights[param] = weights[param] + sigma * jitter
    return new_weights


def compute_fitness(w, target_net, ):
    target_net.load_state_dict(w)
    rewards, _ = play_game(50, Game, CNNPlayer, target_net)
    return sum(rewards)
    

def optimize_model(target_net, fitnesses):
    npop = 20    # population size
    num_episodes = 1
    sigma = 0.1    # noise standard deviation
    alpha = 0.0001  # learning rate
    w = target_net.state_dict()
    for i in range(num_episodes):
        R = np.zeros(npop)
        jitters = {}
        for param in w:
            jitters[param] = []
        for j in range(npop):
            w_try = update(w, sigma, jitters)
            R[j] = compute_fitness(w_try,  target_net)
        print(max(R))
        if np.sum(R) != 0 and np.std(R)!= 0:
            print("here")
            fitnesses.append(np.sum(R))
            A = (R - np.mean(R)) / np.std(R)
            for param in w:
                N = torch.stack(jitters[param])
                w[param] = w[param] + (alpha/(npop*sigma) * np.dot(N.T, A)).T



def train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file, round=100):
    target_net = load_net(initial_checkpoint, 65, CNNSimple)
    fitnesses = []
    for _ in range(round):
        optimize_model(target_net, fitnesses)
    torch.save({
                'state_dict': target_net.state_dict(),
            }, selfplay_checkpoint)
    plt.clf()
    plt.plot(range(len(fitnesses)), fitnesses)
    plt.savefig(fitness_file)

if __name__ == '__main__':
    """
    TEST 1: with prior memory and checkpoint 
    """
    initial_checkpoint = "es_sl_checkpoint.pth.tar"
    selfplay_checkpoint ="es_selfplay_sl.tar"
    record_file = "es_sl_record.json"
    memory_file = "es_sl_memory.json"
    fitness_file = "es_sl_fitness.pdf"
    train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file)

    # """
    # TEST 2: without prior memory and checkpoint 
    # """
    # initial_checkpoint = None
    # selfplay_checkpoint ="es_selfplay.tar"
    # record_file = "es_record.json"
    # memory_file = "es_memory.json"
    # fitness_file = "es_fitness.pdf"
    # train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file)
