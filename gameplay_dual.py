from solitaire.game import SolitaireGame
from players.random_player import RandomPlayer
from players.greedy_player import GreedyPlayer
from players.frugal_player import FrugalPlayer
from utils.dqn_utils import plot_rewards_losses
from utils.game_utils import check_win

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


destination_cards_a = [
    (1, 3),
    (1, 6)
]

destination_cards_b = [
    (0, 6),
    (3, 6)
]


def play_game(iterations, game_class, player_class_a, player_class_b, model=None, update=False):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    winners = []
    rounds = []
    trains_a = []
    trains_b = []

    for _ in range(iterations):
        np.random.shuffle(deck_cards) 
        game = game_class(num_vertices, num_route_colors, edges, deck_cards)
        player_a = player_class_a(num_card_colors, destination_cards_a, 10, model, 1)
        player_b = player_class_b(num_card_colors, destination_cards_b, 10, model, 2)
        game.draw_cards(player_a)
        game.draw_cards(player_a)
        game.draw_cards(player_b)
        game.draw_cards(player_b)

        player_index = 0
        players = [player_a, player_b]
        round = 0
        while check_win(game, players) == -1:
            player = players[player_index]
            if player.draw_or_claim(game) == 0:
                game.draw_cards(player)
            else:
                route = player.choose_route(game)
                current_status = deepcopy(game)
                current_cards = player.cards[:]
                game.claim_route(route, player)

            player_index = (player_index + 1) % 2
            round += 1

        
        winners.append(check_win(game, players))
        rounds.append(round)
        trains_a.append(player_a.trains)
        trains_b.append(player_b.trains)

        # print(player.cards)
        # print(player_a.routes, player_b.routes)
    return winners, trains_a, trains_b, rounds
    
    
winners, trains_a, trains_b, rounds = play_game(1000, SolitaireGame, FrugalPlayer, FrugalPlayer)
# print(winners, trains_a, trains_b)
fig, ax = plt.subplots(4)
ax[0].hist(trains_a, density=False, bins=8)
ax[0].title.set_text("A trains left")
ax[1].hist(trains_b, density=False, bins=8)
ax[1].title.set_text("B trains left")
ax[2].hist(rounds, density=False, bins=10)
ax[2].title.set_text("number of rounds")
ax[3].hist(winners, density=False, bins=3,)
ax[3].title.set_text("winner")
fig.tight_layout()
plt.show()





