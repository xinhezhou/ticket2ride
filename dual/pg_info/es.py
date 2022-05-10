from re import T
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from dual import play_game
sys.path.append("../../")
from RLplayers.pg_player import PGNetwork, PGPlayer
from nonRLplayers.greedy_player import GreedyPlayer
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


def compute_fitness(w, target_net):
    target_net.load_state_dict(w)
    rewards_first, _ = play_game(50, Game, [PGPlayer, GreedyPlayer], models=[target_net, target_net], deck=None, hide_cards=[True, True])
    rewards_second, _ = play_game(50, Game, [GreedyPlayer, PGPlayer], models=[target_net, target_net], deck=None, hide_cards=[True, True])
    return (sum(rewards_first[0])/len(rewards_first[0]) + sum(rewards_second[1])/len(rewards_second[1])) / 2
    

def optimize_model(target_net, average_fitnesses, max_fitnesses, sigma):
    npop = 10    # population size
    # need to compute the average and variance of all terms in parameter
    w = target_net.state_dict()
    R = np.zeros(npop)
    parameters = []
    jitters = {}
    for param in w:
        jitters[param] = []
    for j in range(npop):
        w_try = update(w, sigma, jitters)
        R[j] = compute_fitness(w_try,  target_net)
        parameters.append(w_try)
    average_fitnesses.append(sum(R)/len(R))
    max_fitnesses.append(max(R))
    target_net.load_state_dict(parameters[np.argmax(R)])




def train_es_selfplay(initial_checkpoint, selfplay_checkpoint, average_fitness_file, max_fitness_file, round=1):
    sigma = 0.1
    target_net = load_net(initial_checkpoint, 874, PGNetwork)
    average_fitnesses = []
    max_fitnesses = []
    for _ in range(round):
        optimize_model(target_net, average_fitnesses, max_fitnesses, sigma)
        if _ % 10 == 0:
            print(average_fitnesses[-1], max_fitnesses[-1])
        sigma *= 0.999
    torch.save({
                'state_dict': target_net.state_dict(),
            }, selfplay_checkpoint)
    if average_fitness_file is not None:
        plt.clf()
        plt.plot(range(len(average_fitnesses)), average_fitnesses)
        plt.savefig(average_fitness_file)
        plt.clf()
        plt.plot(range(len(max_fitnesses)), max_fitnesses)
        plt.savefig(max_fitness_file)

if __name__ == '__main__':
    """
    TEST 1: with prior memory and checkpoint 
    """
    initial_checkpoint = None
    selfplay_checkpoint ="es_supervised_model.pth.tar"
    average_fitness_file = "es_supervised_average_fitness.pdf"
    max_fitness_file = "es_supervised_max_fitness.pdf"
    train_es_selfplay(initial_checkpoint, selfplay_checkpoint, average_fitness_file, max_fitness_file, round=200)

    # """
    # TEST 2: without prior memory and checkpoint 
    # """
    # initial_checkpoint = None
    # selfplay_checkpoint ="es_selfplay.tar"
    # record_file = "es_record.json"
    # memory_file = "es_memory.json"
    # fitness_file = "es_fitness.pdf"
    # train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file)