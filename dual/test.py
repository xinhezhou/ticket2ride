from bot_gameplay import play_game
from game import Game
from nonRLplayers.greedy_player import GreedyPlayer
from nonRLplayers.random_player import RandomPlayer

import numpy as np
import json

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
    player_a = GreedyPlayer(num_card_colors, destination_cards_a, 10, 1)
    player_b = RandomPlayer(num_card_colors, destination_cards_b, 10, 2)
    game.draw_cards(player_a)
    game.draw_cards(player_a)
    game.draw_cards(player_b)
    game.draw_cards(player_b)
    players = [player_a, player_b]
    player_a.players = players

    return game, players

winners, records = play_game(10000, initialize_game)
with open("greedy_random_more.json", "w") as outfile:
    json.dump(records, outfile)
print(winners)
