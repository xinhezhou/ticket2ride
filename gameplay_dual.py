from solitaire.game import SolitaireGame
from players.random_player import RandomPlayer
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
destination_cards_a = [
    (0, 6),
]

destination_cards_b = [
    (1, 5),
]


def play_game(iterations, game_class, player_class_a, player_class_b, model=None, update=False):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    winners = []
    losses = []
    rewards = []

    for _ in range(iterations):
        np.random.shuffle(deck_cards) 
        game = game_class(num_vertices, num_route_colors, edges, deck_cards)
        player_a = player_class_a(num_card_colors, destination_cards_a, model, 1)
        player_b = player_class_b(num_card_colors, destination_cards_b, model, 2)
        game.draw_cards(player_a)
        game.draw_cards(player_a)
        game.draw_cards(player_b)
        game.draw_cards(player_b)

        player_index = 0
        players = [player_a, player_b]
        while check_win(game.status, players) == -1:
            player = players[player_index]
            if player.draw_or_claim(game.graph, game.status) == 0:
                if game.card_index < len(game.cards):
                    game.draw_cards(player)
                else:
                    print("no")
                    break
            else:
                route = player.choose_route(game.graph, game.status)
                current_status = deepcopy(game.status)
                current_cards = player.cards[:]
                game.claim_route(route, player)
                if update:
                    reward, loss = player.update_model(game.graph, current_status, game.status, current_cards, player.cards, route)
                    losses.append(loss)
                    rewards.append(reward)

            player_index = (player_index + 1) % 2
        
        winners.append(check_win(game.status, players))
        # print(player.cards)
        # print(player.routes)
    if update:
        return winners
    else:
        return winners


winners = play_game(10, SolitaireGame, RandomPlayer, RandomPlayer)
print(winners)
# plt.show()





