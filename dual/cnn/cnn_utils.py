

from exprience_replay import Transition
from mcts_utils import simulate_rollout
import torch.nn as nn
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
import json

from utils.game_utils import compute_availability_matrix, get_available_routes
sys.path.append("../")
from bot_gameplay import play_game
sys.path.append("../")
from game import Game
from nonRLplayers.random_player import RandomPlayer
# from cnn_player import CNNPlayer




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

def generate_gameplay(records):
    """
    Generate gameplays from records and push them into memory
    for CNN training
    """
    num_vertices = 7
    num_route_colors = 7
    num_card_colors = 7
    edges = {
        (0, 1, 2): 1,
        (0, 3, 2): 3,
        (0, 3, 5): 3,
        (0, 4, 6): 2,
        (1, 2, 6): 2,
        (1, 4, 1): 2,
        (1, 4, 2): 2,
        (2, 4, 1): 1,
        (2, 4, 3): 1,
        (2, 6, 6): 2,
        (3, 4, 4): 2,
        (3, 5, 4): 1,
        (3, 5, 6): 1,
        (4, 5, 3): 2,
        (4, 5, 6): 2,
        (4, 6, 6): 3,
        (5, 6, 2): 3,
        (5, 6, 3): 3,
    }
    memory = {}
    for key in records:
        record = records[key]
        deck_cards = record["deck"]
        game = Game(num_vertices, num_route_colors, edges, deck_cards)
        player_a = RandomPlayer(num_card_colors, record["destinations_a"], 10, 1)
        player_b = RandomPlayer(num_card_colors, record["destinations_b"], 10, 2)
        players = [player_a, player_b]
        for action in record["actions"]:
            reward = 0
            state = generate_state_matrix(game, players).unsqueeze(0)
            player = players[0]
            if len(action) == 2:
                choice = 0
                game.draw_cards(player)
            else:
                u, v, c = action
                choice = 1 + u*49 + v*7 + c
                game.claim_route(action, player)
                if action == record["actions"][-1]:
                    reward = 1
            next_state = generate_state_matrix(game, players).unsqueeze(0)
            if player.id == 1:
                entry = {}
                entry["state"]  = state.tolist()
                entry["choice"] = [[choice]]
                entry["next_state"] = next_state.tolist()
                entry["reward"] = [reward]
                memory[key] = entry
            players = players[::-1]
    return memory
            

def optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    next_state_batch = torch.cat(batch.next_state)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(action_batch, reward_batch)
    action_probs = policy_net(state_batch)
    state_action_values = action_probs.gather(1, action_batch)
    

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # print(state_action_values, expected_state_action_values)
    entropy = 0
    # print(action_probs)
    for i in range(action_probs.shape[0]):
        probs_list = action_probs[i][action_probs[i] != 0]
        entropy -= torch.sum(probs_list * torch.log(probs_list))
        if torch.isnan(entropy):
            print("what")
            print(probs_list, torch.log(probs_list))
            return 
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) + 0.0001 * entropy
    # print(entropy)
    # entropy = np.sum(np.mean(action_probs) * np.log(action_probs))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def compute_route_freq(records, k):
    route_freq = {}
    win_freq = {}
    for key in records:
        record = records[key]
        actions = record["actions"]
        winner = tuple(record["winner"])
        if winner in win_freq:
            win_freq[winner] += 1
        else:
            win_freq[winner] = 1
        # print(actions[0])
        routes = []
        for i in range(0, len(actions), 2):
            action = actions[i]
            if len(action) == 3:
                routes.append(tuple(action[:2]))
        routes.sort()
        if tuple(routes) in route_freq:
            route_freq[tuple(routes)] += 1
        else:
            route_freq[tuple(routes)] = 1

    for key in route_freq:
        if route_freq[key] > k:
            print(key, route_freq[key])
    print(sorted(route_freq.values()))
    print(win_freq)
    return win_freq, route_freq


def convert_records(json_file, memory_file):
    f = open(json_file)
    records = json.load(f)
    records = generate_gameplay(records)
    with open(memory_file, "w") as outfile:
        json.dump(records, outfile)


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

def evaluate_net(target_net, net_file, record_file):
    def initialize_game():
        num_vertices = 7
        num_route_colors = 7
        num_card_colors = 7
        deck_cards = [1,2,3,4,5,6,0] * 10  + [0] * 2 # train cards in the deck
        edges = {
        (0, 1, 2): 1,
        (0, 3, 2): 3,
        (0, 3, 5): 3,
        (0, 4, 6): 2,
        (1, 2, 6): 2,
        (1, 4, 1): 2,
        (1, 4, 2): 2,
        (2, 4, 1): 1,
        (2, 4, 3): 1,
        (2, 6, 6): 2,
        (3, 4, 4): 2,
        (3, 5, 4): 1,
        (3, 5, 6): 1,
        (4, 5, 3): 2,
        (4, 5, 6): 2,
        (4, 6, 6): 3,
        (5, 6, 2): 3,
        (5, 6, 3): 3,
        }

        destinations = [
            (1, 5),
            (1, 3),
            (3, 6),
            (1, 6),
            (0, 6)
        ]

        np.random.shuffle(deck_cards) 
        game = Game(num_vertices, num_route_colors, edges, deck_cards)
        # sample = np.random.choice(5, 4, False)
        destination_cards_a = [(1, 3), (1, 6)]
        destination_cards_b = [(0, 6), (3, 6)]
        player_a = CNNPlayer(num_card_colors, destination_cards_a, 10, 1, target_net)
        # player_b = GreedyPlayer(num_card_colors, destination_cards_b, 10, 2,)
        player_b = RandomPlayer(num_card_colors, destination_cards_b, 10, 2,)
        game.draw_cards(player_a)
        game.draw_cards(player_a)
        game.draw_cards(player_b)
        game.draw_cards(player_b)
        players = [player_a, player_b]

        return game, players
    checkpoint = torch.load(net_file)
    target_net.load_state_dict(checkpoint['state_dict'])
    winners, records = play_game(10000, initialize_game)
    print(winners)
    with open(record_file, "w") as outfile:
        json.dump(records, outfile)
    plt.hist(winners, density=False, bins=3,)
    plt.show()
