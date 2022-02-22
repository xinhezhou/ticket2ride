
from exprience_replay import ReplayMemory
import json
from cnn_utils import generate_gameplay, optimize_model, compute_route_freq, evaluate_net, push_memory, convert_records
from cnn_network import Network
from cnn_player import CNNPlayer
# from test_player import TestPlayer
import torch.optim as optim
import torch
import numpy as np
import sys
sys.path.append("..")
from bot_gameplay import play_game
from game import Game
from nonRLplayers.random_player import RandomPlayer
from nonRLplayers.greedy_player import GreedyPlayer
import matplotlib.pyplot as plt



# f = open("../greedy_random_shuffled.json")
# records = json.load(f)
# memory = generate_gameplay(records)
# with open("greedy_random_lookahead_memory.json", "w") as outfile:
#     json.dump(memory, outfile)



# memory = ReplayMemory(100000)
# f = open("greedy_random_lookahead_memory.json")
# records = json.load(f)
# for key in records:
#     record = records[key]
#     state = torch.tensor(record["state"])
#     choice = torch.tensor(record["choice"])
#     next_state = torch.tensor(record["next_state"])
#     reward = torch.tensor(record["reward"])
#     memory.push(state, choice, next_state, reward)
# for _ in range(1000):
#     for i in range(10):
#         optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
#     target_net.load_state_dict(policy_net.state_dict())
# torch.save({
#             'state_dict': target_net.state_dict(),
#         }, "greedy_random_lookahead.pth.tar")
# plt.plot(range(len(losses)), losses)
# plt.show()

def evaluate_net(target_net, net_file, record_file, counter=1000, display=True):
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
    winners, records = play_game(counter, initialize_game)
    print(winners)
    with open(record_file, "w") as outfile:
        json.dump(records, outfile)
    if display:
        plt.hist(winners, density=False, bins=3,)
        plt.show()


policy_net = Network((96, 7, 7), 7*7*7+1, 32)
target_net = Network((96, 7, 7), 7*7*7+1, 32)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
BATCH_SIZE = 128
GAMMA = 0.9
TARGET_UPDATE = 10
losses = []

memory = ReplayMemory(100000)
push_memory("greedy_random_lookahead_memory.json", memory)

# for _ in range(1000):
#     for i in range(10):
#         optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
#     target_net.load_state_dict(policy_net.state_dict())
# torch.save({
#             'state_dict': target_net.state_dict(),
#         }, "greedy_random_selfplay.pth.tar")

# evaluate_net(target_net, net_file="greedy_random_lookahead.pth.tar", record_file="cnn_random_selfplay.json", counter=1000, display=True)

# checkpoint = torch.load("greedy_random_lookahead.pth.tar")
# target_net.load_state_dict(checkpoint['state_dict'])
# policy_net.load_state_dict(checkpoint['state_dict'])
# for _ in range(10):
#     convert_records("cnn_random_selfplay.json", "cnn_random_selfplay_memory.json")
#     push_memory("cnn_random_selfplay_memory.json", memory)
#     for i in range(10):
#         optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
#     target_net.load_state_dict(policy_net.state_dict())
#     torch.save({
#                 'state_dict': target_net.state_dict(),
#             }, "greedy_random_selfplay.pth.tar")
#     evaluate_net(target_net, net_file="greedy_random_selfplay.pth.tar", record_file="cnn_random_selfplay.json", counter=100, display=True)

# for _ in range(10):
#     evaluate_net(target_net, net_file="greedy_random_selfplay.pth.tar", record_file="cnn_random_selfplay.json", counter=100, display=False)
#     convert_records("cnn_random_selfplay.json", "cnn_random_selfplay_memory.json")
#     push_memory("cnn_random_selfplay_memory.json", memory)
#     checkpoint = torch.load("greedy_random_selfplay.pth.tar")
#     target_net.load_state_dict(checkpoint['state_dict'])
#     policy_net.load_state_dict(checkpoint['state_dict'])
#     for i in range(10):
#         optimize_model(policy_net, target_net, optimizer, memory, losses, BATCH_SIZE, GAMMA)
#         target_net.load_state_dict(policy_net.state_dict())
#     torch.save({
#             'state_dict': target_net.state_dict(),
#         }, "greedy_random_selfplay.pth.tar")

# evaluate_net(target_net, net_file="greedy_random_selfplay.pth.tar", record_file="cnn_random_selfplay.json", counter=10, display=False)
# checkpoint = torch.load("greedy_random_lookahead.pth.tar")
# target_net.load_state_dict(checkpoint['state_dict'])

# winners, records = play_game(1000, initialize_game)
# print(winners)
# with open("cnn_random_lookahead.json", "w") as outfile:
#     json.dump(records, outfile)
# plt.hist(winners, density=False, bins=3,)
# plt.show()

f = open("cnn_random_selfplay.json")
records = json.load(f)
win_freqs, route_freqs = compute_route_freq(records, 1)