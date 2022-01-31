
from exprience_replay import ReplayMemory
import json
from cnn_utils import generate_gameplay, optimize_model
from cnn_network import Network
from cnn_player import CNNPlayer
from test_player import TestPlayer
import torch.optim as optim
import torch
import numpy as np
import sys
sys.path.append("..")
from bot_gameplay import play_game
from game import Game
from nonRLplayers.random_player import RandomPlayer
import matplotlib.pyplot as plt

policy_net = Network((160, 7, 7), 7*7*7+1, 32)
target_net = Network((160, 7, 7), 7*7*7+1, 32)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.0001)
BATCH_SIZE = 10
GAMMA = 0.9
losses = []

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

    # np.random.shuffle(deck_cards) 
    game = Game(num_vertices, num_route_colors, edges, deck_cards)
    # sample = np.random.choice(5, 4, False)
    destination_cards_a = [(1, 3), (1, 6)]
    destination_cards_b = [(0, 6), (3, 6)]
    player_a = CNNPlayer(num_card_colors, destination_cards_a, 10, 2, target_net)
    player_b = RandomPlayer(num_card_colors, destination_cards_b, 10, 1,)
    game.draw_cards(player_a)
    game.draw_cards(player_a)
    game.draw_cards(player_b)
    game.draw_cards(player_b)
    players = [player_b, player_a]
    player_a.players = players
    player_b.players = players

    return game, players


# memory = ReplayMemory(10000)
# f = open("../greedy_random_fixed.json")
# records = json.load(f)
# generate_gameplay(records, memory)


# for _ in range(1000):
#     for i in range(10):
#         optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
#     target_net.load_state_dict(policy_net.state_dict())

# torch.save({
#             'state_dict': target_net.state_dict(),
#         }, "greedy_random_fixed.pth.tar")

# plt.plot(range(len(losses)), losses)
# plt.show()



checkpoint = torch.load("greedy_random_fixed.pth.tar")
target_net.load_state_dict(checkpoint['state_dict'])
winners, records = play_game(1000, initialize_game)
print(winners)
plt.hist(winners, density=False, bins=3,)
plt.show()