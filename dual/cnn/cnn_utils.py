

from cnn_network import CNNSimple
from mcts_utils import simulate_rollout
import torch.nn as nn
import torch
import numpy as np
import sys
import json

sys.path.append("../")
sys.path.append("../")
from utils.game_utils import compute_availability_matrix





def generate_multigraph_matrix(initial_matrix, max_count):
    """
    Generate a binary matrix of shape D * V * V, where D = max_count * C.
    This matrix concatenated from max_val binary matrices of shape 
    C * V * V where each matrix corresponds to entires of a particular 
    value in range [1, max_count] in the initial matrix
    """
    matrix_list = []
    u, v, c = initial_matrix.shape
    for i in range(1, max_count+1):
        matrix = torch.zeros((c, u, v))
        for x in range(u):
            for y in range(v):
                for z in range(c):
                    if initial_matrix[x][y][z] == i:
                        matrix[z][x][y] = 1
        matrix_list.append(matrix)
    return torch.cat(matrix_list)

def generate_destination_cards_matrix(destination_cards, v):
    """
    Generate a binary matrix of shape 1 * V * V, where each entry
    (i, j) represents whether there is a destination card between
    i and j
    """
    matrix = torch.zeros((1, v, v))
    for i, j in destination_cards:
        matrix[0][i][j] = 1
    return matrix
    
def generate_deck_matrix(deck, card_index, v):
    """
    Generate a binary matrix of shape len(deck) * V * V, where
    row c in the i_th channel represents the i_th card is of
    color c
    """
    matrix = torch.zeros((len(deck), v, v))
    for i in range(card_index, len(deck)):
        c = deck[i]
        for j in range(v):
            matrix[i][c][j] = 1
    return matrix

def generate_train_cards_matrix(train_cards, max_count, v):
    """
    Generate a binary matrix of shape max_count * V * V, where
    row c in the i_th channel represents the there are at least 
    i cards of color c in train cards
    """
    matrix = torch.zeros(max_count, v, v)
    for i in range(max_count):
        for j in range(v):
            if j < len(train_cards) and train_cards[j] > i:
                for k in range(v):
                    matrix[i][j][k] = 1
    return matrix

def generate_train_matrix(num_trains, max_count, v):
    """
    Generate a binary matrix of shape max_count * V * V, where
    row c in the i_th channel represents the there are at least 
    i cards of color c in train cards
    """
    matrix = torch.zeros(max_count, v, v)
    for i in range(max_count):
        if num_trains > i:
            for j in range(v):
                for k in range(v):
                    matrix[i][j][k] = 1
    return matrix

def generate_route_lookahead_matrix(game, players):
    u,v,c = game.status.shape
    matrix = torch.zeros((c,u,v))
    availability = compute_availability_matrix(game.graph, game.status, players[0])
    for i in range(u):
        for j in range(v):
            for k in range(c):
                if availability[i][j][k]:
                    outcome = simulate_rollout(game, players, (i, j, k)) + simulate_rollout(game, players, (i, j, k)) + simulate_rollout(game, players, (i, j, k))
                    if outcome[0] == 3:
                        matrix[k][i][j] = 1
    return matrix

def generate_card_lookahead_matrix(game, players):
    outcome = simulate_rollout(game, players, 0) + simulate_rollout(game, players, 0) + simulate_rollout(game, players, 0)
    u,v,c = game.status.shape
    if outcome[0] == 3:
        return torch.ones((1, u,v))
    else:
        return torch.zeros((1, u,v))

def generate_state_matrix(game, players):
    """
    generate a binary matrix that represents the current state of game
    for the input of the CNN
    """
    matrices = []
    v = len(game.graph)
    matrices.append(generate_multigraph_matrix(game.graph, 4)) # 28 * 7 * 7
    matrices.append(generate_multigraph_matrix(game.status, 2)) # 14  * 7 * 7
    # matrices.append(generate_deck_matrix(game.cards, game.card_index, v)) # 72 * 7 * 7
    matrices.append(generate_route_lookahead_matrix(game, players)) # 7 * 7 * 7
    matrices.append(generate_card_lookahead_matrix(game, players)) # 1 * 7 * 7
    for player in players:
        matrices.append(generate_destination_cards_matrix(player.destination_cards, v)) # 2 * 7 * 7
        matrices.append(generate_train_cards_matrix(player.cards, 12, v)) # 24 * 7 * 7
        matrices.append(generate_train_matrix(player.trains, 10, v)) # 0 * 7 * 7

    return torch.cat(matrices)

def load_net(checkpoint_file, network=CNNSimple, eval=False):
    net = network((96, 7, 7), 7*7*7+1, 32)
    if eval:
        net.eval()
    if checkpoint_file is not None:

        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['state_dict'])
    return net

def push_memory(memory_file, memory):
    f = open(memory_file)
    records = json.load(f)
    for key in records:
        record = records[key]
        state = torch.tensor(record["state"])
        choice = torch.tensor(record["choice"])
        next_state = torch.tensor(record["next_state"])
        reward = torch.tensor(record["reward"])
        memory.push(state, choice, next_state, reward)
