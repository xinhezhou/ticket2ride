
import numpy as np
import json
import sys
sys.path.append("..")
from game import Game
from nonRLplayers.random_player import RandomPlayer
from nonRLplayers.greedy_player import GreedyPlayer
from nonRLplayers.frugal_player import FrugalPlayer
from RLplayers.cnn_player import CNNPlayer
from RLplayers.rl_utils import load_net
from RLplayers.cnn_network import CNNSimple
from utils.game_utils import compute_availability_matrix, get_available_routes

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
    (1, 5)
]

def check_win(player, game):
    if len(player.destination_cards) == 0:
        return 2
    elif player.trains == 0 and len(player.destination_cards) == 1:
        return 1
    elif player.trains == 0 and len(player.destination_cards) == 2:
        return 0
    elif game.card_index > len(game.cards)-3:
        return -1
    return False

def play_game(iterations, game_class, player_class, eps=0, model=None, update=False):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    trains = []
    rounds = []
    losses = []
    rewards = []
    records = {}
    for i in range(iterations):
        np.random.shuffle(deck_cards) 
        game = game_class(num_vertices, num_route_colors, edges, deck_cards)
        player = player_class(num_card_colors, destination_cards, 10, 1, model)
        game.draw_cards(player)
        game.draw_cards(player)
        record = {}
        record["deck"] = game.cards
        record["destinations"] = player.destination_cards
        actions = []
        while check_win(player, game) is False:
            availability = compute_availability_matrix(game.graph, game.status, player)
            available_routes = get_available_routes(availability)
            if np.random.random() < eps:
                action = np.random.randint(len(available_routes)+1)
                if action == 0:
                    cards = game.draw_cards(player)
                    actions.append(cards)
                else:
                    route = available_routes[action-1]
                    game.claim_route(route, player)
                    actions.append(route)

            elif len(available_routes) == 0 or player.draw_or_claim(game, [player]) == 0:
                cards = game.draw_cards(player)
                actions.append(cards)
            else:
                route = player.choose_route(game, [player])
                if route[0] == 0 and route[1] == 0:
                    print(available_routes)
                game.claim_route(route, player)
                actions.append(route)

        reward = check_win(player, game)
        record["actions"] = actions
        record["reward"] = reward
        rewards.append(reward)
        records[i] = record
    return rewards, records


        # rounds.append(num_rounds)
        # trains.append(player.trains_used)
        # print(player.cards)
        # print(player.routes)
    # if update:
    #     return rewards, losses
    # else:
    #     return trains, rounds


if __name__ == '__main__':
    # rewards, records = play_game(1000, Game, RandomPlayer)
    # print(sum(rewards))


    # rewards, records = play_game(1000, Game, GreedyPlayer)
    # with open("dqn_record.json", "w") as outfile:
    #     json.dump(records, outfile)
    # print(sum(rewards), records)

    # rewards, records = play_game(100, Game, FrugalPlayer)
    # print(sum(rewards))


    # net_file = None
    net_file = "dqn_selfplay.pth.tar"
    model = load_net(net_file, 65, CNNSimple, eval=True)
    rewards, records = play_game(100, Game, CNNPlayer, model)
    print(sum(rewards))
    # print(records)

