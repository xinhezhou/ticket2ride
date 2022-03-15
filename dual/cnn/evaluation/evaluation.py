
import numpy as np
import json 
import sys
import matplotlib.pyplot as plt
sys.path.append("../")
from cnn_player import CNNPlayer
from cnn_utils import load_net
from cnn_network import CNNSimple, CNNComplex
sys.path.append("../../")
from bot_gameplay import play_game
sys.path.append("../")
from game import Game
from nonRLplayers.random_player import RandomPlayer
from nonRLplayers.greedy_player import GreedyPlayer


def generate_players(player_classes, nets, destination_cards):
    num_card_colors = 7
    player_a = player_classes[0](num_card_colors, destination_cards[0], 10, 1, nets[0])
    player_b = player_classes[1](num_card_colors, destination_cards[1], 10, 2, nets[1])
    return [player_a, player_b]



def evaluate_net(target_nets, player_classes, destination_cards, record_file, counter=100, display=True):
    def initialize_game():
        num_vertices = 7
        num_route_colors = 7
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

        np.random.shuffle(deck_cards) 
        game = Game(num_vertices, num_route_colors, edges, deck_cards)
        players = generate_players(player_classes, target_nets, destination_cards)
        player_a, player_b = players 
        game.draw_cards(player_a)
        game.draw_cards(player_a)
        game.draw_cards(player_b)
        game.draw_cards(player_b)
        
        return game, players
    
    winners, records = play_game(counter, initialize_game)
    print(winners)
    if record_file is not None:
        with open(record_file, "w") as outfile:
            json.dump(records, outfile)
    if display:
        plt.hist(winners, density=False, bins=3,)
        plt.show()
    return winners, records

if __name__ == '__main__':

    filenames = [
    ["../rl/with_prior_es.pth.tar", "with_prior_es_random.json"],
    ["../rl/without_prior_es.pth.tar", "without_prior_es_random.json"],
    ]
    for net_file, record_file in filenames:
        target_nets = [
            load_net(net_file,CNNSimple, eval=True),
            load_net(None, eval=True),
        ]
        player_classes = [CNNPlayer, RandomPlayer]
        destination_cards = [
            [[1, 3], [1, 6]],
            [[0, 6], [3, 6]],
        ]
        evaluate_net(target_nets, player_classes, destination_cards, record_file, counter=1000, display=False)
    
