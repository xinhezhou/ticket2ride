from game import SolitaireGame
from players.random_player import RandomPlayer
# from utils.game_utils import check_path, compute_availability_matrix
# from utils.dqn_utils import plot
# from utils.translate_utils import translate_route

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# import torch
# import torch.nn as nn
# import torch.optim as optim

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
destination_cards = [
    (0, 6),
]

#########################
###    DQN Setup     ####
#########################
# num_inputs = (num_vertices * num_vertices * num_route_colors) * 2 +  + num_card_colors
# num_outputs = num_vertices * num_vertices * num_route_colors
# policy_net = Network(num_inputs, num_outputs, 100, "cpu")
# model = {
#     "net": policy_net,
#     "loss_fn": nn.SmoothL1Loss(),
#     "optimizer":optim.Adam(policy_net.parameters()),
#     "gamma": 0.9,
# }


def play_game(iterations, game_class, player_class, model=None, update=False):
    trains = []
    rounds = []
    losses = []
    rewards = []

    for _ in range(iterations):
        np.random.shuffle(deck_cards) 
        game = game_class(num_vertices, num_route_colors, edges, deck_cards)
        player = player_class(num_card_colors, destination_cards)
        game.draw_cards(player)
        game.draw_cards(player)
        num_rounds = 0
        while len(player.destination_cards) > 0:
            num_rounds += 1
            if player.draw_or_claim(game.graph, game.status) == 0:
                if game.card_index < len(game.cards):
                    game.draw_cards(player)
                else:
                    print("heck")
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
        

        rounds.append(num_rounds)
        trains.append(player.trains_used)
        # print(player.cards)
        # print(player.routes)
    if update:
        return rewards, losses
    else:
        return trains, rounds


# rewards, losses = play_game(1000, True)
# print(rewards)
# print(losses)
# fig, ax = plt.subplots(2)
# plot(rewards, losses, ax, 50)
# plt.savefig("diagrams/solitaire_dqn_loss.pdf")


trains, rounds = play_game(1000, SolitaireGame, RandomPlayer)
print(trains)
print(rounds)


fig, ax = plt.subplots(2)
ax[0].hist(trains, density=False, bins=8)
ax[0].title.set_text("trains used")
ax[1].hist(rounds, density=False, bins=10, range=(0,50))
ax[1].title.set_text("number of rounds")
fig.tight_layout()
# plt.savefig("diagrams/solitaire_dqn.pdf")
plt.show()





