from game import Game
import json
from players.random_player import RandomPlayer
from players.greedy_player import GreedyPlayer
from players.frugal_player import FrugalPlayer
from utils.dqn_utils import plot_rewards_losses
from utils.game_utils import check_win, compute_availability_matrix, get_available_routes

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim

#########################
###    Game Setup    ####
#########################
num_vertices = 7
num_route_colors = 7
num_card_colors = 7
deck_cards = [0] * 12  + [1,2,3,4,5,6] * 10 # train cards in the deck
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




def initialize_game():
    game = Game(num_vertices, num_route_colors, edges, deck_cards)
    destination_cards_a = [
    (1, 3),
    (1, 6)
    ]

    destination_cards_b = [
        (0, 6),
        (3, 6)
    ]
    player_a = GreedyPlayer(num_card_colors, destination_cards_a, 10, 1)
    player_b = GreedyPlayer(num_card_colors, destination_cards_b, 10, 2)
    game.draw_cards(player_a)
    game.draw_cards(player_a)
    game.draw_cards(player_b)
    game.draw_cards(player_b)
    players = [player_a, player_b]

    return game, players


def play_game(iterations, logging=True):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    winners = []
    records = {}

    for i in range(iterations):
        record = {}
        np.random.shuffle(deck_cards) 
        record["deck"] = deck_cards

        
        player_index = 0
        game, players = initialize_game()
        actions = []
        winner = -1
        while winner == -1:
            player = players[player_index]
            availability = compute_availability_matrix(game.graph, game.status, player)
            if len(get_available_routes(availability)) == 0 or player.draw_or_claim(game) == 0:
                cards = game.draw_cards(player)
                actions.append(cards)
            else:
                route = player.choose_route(game) 
                game.claim_route(route, player)
                actions.append(route)

            player_index = (player_index + 1) % 2
            winner = check_win(game, players)

        record["actions"] = actions
        record["winner"] = winner
        records[i] = record



    
    return winners, records
    
    
winners, records = play_game(1000)
# print(records)
with open("greedy_greedy.json", "w") as outfile:
    json.dump(records, outfile)
# # print(winners, trains_a, trains_b)
# fig, ax = plt.subplots(4)
# ax[0].hist(trains_a, density=False, bins=8)
# ax[0].title.set_text("A trains left")
# ax[1].hist(trains_b, density=False, bins=8)
# ax[1].title.set_text("B trains left")
# ax[2].hist(rounds, density=False, bins=10)
# ax[2].title.set_text("number of rounds")
# ax[3].hist(winners, density=False, bins=3,)
# ax[3].title.set_text("winner")
# fig.tight_layout()
# plt.show()





