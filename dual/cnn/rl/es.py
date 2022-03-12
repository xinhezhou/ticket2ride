import numpy as np
import torch
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from evaluation.evaluation import evaluate_net
from cnn_utils import load_net
from cnn_player import CNNPlayer
from nonRLplayers.random_player import RandomPlayer

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


def compute_fitness(w, target_net, player_classes, opponent_net, destination_cards, first):
    target_net.load_state_dict(w)
    target_nets = [target_net, load_net(opponent_net)]
    if not first:
        target_nets = target_nets[:]
    winners, _ = evaluate_net(target_nets, player_classes, destination_cards, None, counter=100, display=False)
    fitness = 0
    for winner in winners:
        if winner == 0:
            fitness += 0.5
        elif winner == 1 and first:
            fitness += 1
        elif winner == 2 and not first:
            fitness += 1
    return fitness
    

def optimize_model(target_net, fitnesses, player_classes, destination_cards, opponent_net=None,first=True):
    npop = 3    # population size
    num_episodes = 1
    sigma = 0.1    # noise standard deviation
    alpha = 0.001  # learning rate
    w = target_net.state_dict()
    for i in range(num_episodes):
        R = np.zeros(npop)
        jitters = {}
        for param in w:
            jitters[param] = []
        for j in range(npop):
            w_try = update(w, sigma, jitters)
            R[j] = compute_fitness(w_try,  target_net, player_classes, opponent_net, destination_cards, first)
        if np.sum(R) != 0 and np.std(R)!= 0:
            print("here")
            fitnesses.append(np.sum(R))
            A = (R - np.mean(R)) / np.std(R)
            for param in w:
                N = torch.stack(jitters[param])
                w[param] = w[param] + (alpha/(npop*sigma) * np.dot(N.T, A)).T



def train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file, player_classes, destination_cards, round=1000):
    target_net = load_net(initial_checkpoint)
    fitnesses = []
    for _ in range(round):
        optimize_model(target_net, fitnesses, player_classes, destination_cards)
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
    # initial_checkpoint = "../sl/medium_m.pth.tar"
    # selfplay_checkpoint ="with_prior_es.pth.tar"
    # record_file = "../dataset/with_prior_es.json"
    # memory_file = "../dataset/with_prior_es_memory.json"
    # fitness_file = "with_prior_es.pdf"
    # player_classes = [
    #     CNNPlayer, RandomPlayer
    # ]
    # destination_cards = [
    #     [[1, 3], [1, 6]],
    #     [[0, 6], [3, 6]],
    # ]
    # train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file, player_classes, destination_cards)

    """
    TEST 2: without prior memory and checkpoint 
    """
    initial_checkpoint = None
    selfplay_checkpoint ="without_prior_es.pth.tar"
    record_file = "../dataset/without_prior_es.json"
    memory_file = "../dataset/without_prior_es_memory.json"
    fitness_file = "without_prior_es.pdf"
    player_classes = [
        CNNPlayer, RandomPlayer
    ]
    destination_cards = [
        [[1, 3], [1, 6]],
        [[0, 6], [3, 6]],
    ]
    train_es_selfplay(initial_checkpoint, selfplay_checkpoint, fitness_file, player_classes, destination_cards)
