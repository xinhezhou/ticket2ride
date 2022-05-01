from gc import set_debug
from random import Random
import numpy as np
import json
import sys
sys.path.append("../../")
from game import Game
from nonRLplayers.random_player import RandomPlayer
from nonRLplayers.greedy_player import GreedyPlayer
from RLplayers.pg_player import PGNetwork, PGPlayer
from RLplayers.cnn_player import CNNPlayer
from RLplayers.rl_utils import load_net
from RLplayers.cnn_network import CNNSimple
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress
import matplotlib.pyplot as plt
import pandas as pd


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
    [(1, 3),(1, 6)],
    [(1, 3), (1, 6)]
]

def check_win(players, game):
    for player in players:
        if len(player.destination_cards) == 0 or player.trains == 0:
            return True
    return game.card_index > len(game.cards)-3



def play_game(iterations, game_class, player_classes, eps=0, models=[None,None], update=False, deck = None):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    attacker_rewards =[]
    defender_rewards = []
    records = {}
    deck_cards = [0] * 12  + [1,2,3,4,5,6] * 10 # train cards in the deck
    for i in range(iterations):
        decay = 0.9
        coeff = 1
        if deck is None:
            np.random.shuffle(deck_cards) 
        else:
            deck_cards = deck
        game = game_class(num_vertices, num_route_colors, edges, deck_cards)
        players = []
        for j in range(len(player_classes)):
            player = player_classes[j](num_card_colors, destination_cards[j], 10, j+1, models[j])
            game.draw_cards(player)
            game.draw_cards(player)
            players.append(player)
        record = {}
        record["deck"] = game.cards
        record["destinations"] = destination_cards
        actions = []
        defender_reward = 0
        attacker_reward = 0
        while not check_win(players, game):
            attacker_player = players[0]
            availability = compute_availability_matrix(game.graph, game.status, attacker_player)
            available_routes = get_available_routes(availability)
            if len(available_routes) == 0 or attacker_player.draw_or_claim(game, players) == 0:
                cards = game.draw_cards(attacker_player)
                actions.append(cards)
            else:
                route = attacker_player.choose_route(game, players)
                attacker_reward += compute_progress(game.graph, game.status, route, attacker_player.destination_cards, attacker_player.id) * coeff
                game.claim_route(route, attacker_player)
                actions.append(route)

            
            defender_player = players[1]
            availability = compute_availability_matrix(game.graph, game.status, defender_player)
            available_routes = get_available_routes(availability)
            if len(available_routes) == 0 or defender_player.draw_or_claim(game, players[::-1]) == 0:
                cards = game.draw_cards(defender_player)
                actions.append(cards)
            else:
                route = defender_player.choose_route(game, players[::-1])
                defender_reward += compute_progress(game.graph, game.status, route, defender_player.destination_cards, defender_player.id) * coeff
                game.claim_route(route, defender_player)
                actions.append(route)
            
            coeff *= decay
            

        record["actions"] = actions
        record["defender_reward"] = defender_reward
        record["attacker_reward"] = attacker_reward
        defender_rewards.append(defender_reward)
        attacker_rewards.append(attacker_reward)
        rewards = [attacker_rewards, defender_rewards]
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


def compute_route_freq(records, k, win_file=None):
    route_freq_a = {}
    route_freq_b = {}
    for key in records:
        record = records[key]
        actions = record["actions"]
        # print(actions[0])
        routes_a = []
        routes_b = []
        for i in range(0, len(actions)):
            action = actions[i]
            if len(action) == 3:
                if i % 2 == 0:
                    routes_a.append(tuple(action[:2]))
                else:
                    routes_b.append(tuple(action[:2]))
        routes_a.sort()
        routes_b.sort()
        if tuple(routes_a) in route_freq_a:
            route_freq_a[tuple(routes_a)] += 1
        else:
            route_freq_a[tuple(routes_a)] = 1
        
        if tuple(routes_b) in route_freq_b:
            route_freq_b[tuple(routes_b)] += 1
        else:
            route_freq_b[tuple(routes_b)] = 1

    print("attacker")
    for key in route_freq_a:
        if route_freq_a[key] > k:
            print(key, route_freq_a[key])
    print(sorted(route_freq_a.values()))


    print("defender")
    for key in route_freq_b:
        if route_freq_b[key] > k:
            print(key, route_freq_b[key])
    print(sorted(route_freq_b.values()))
    return route_freq_a, route_freq_b

players = [
    (PGPlayer, "SupervisedPG", "pg_supervised/model.pth.tar")
    # (PGPlayer, "SelfplayPG", "pg_selfplay/model.pth.tar")
    # (PGPlayer, "SupervisedES", "es_supervised/model.pth.tar")
    # (PGPlayer, "SelfplayES", "es_selfplay/model.pth.tar")
    # (RandomPlayer, "Random", None)
    # (GreedyPlayer, "Greedy", None)
]

average_rewards = []

if __name__ == '__main__':
    for player_a in players:
        average_rewards.append([])
        for player_b in players:
            records = {}
            player_a_rewards  = []
            player_b_rewards = []
            player_a_color = u'#1f77b4'
            player_b_color = u'#ff7f0e'

            player_classes = [player_a[0], player_b[0]]
            player_a_label = player_a[1]
            player_b_label = player_b[1]
            player_a_model = load_net(player_a[2], 874, PGNetwork, eval=True)
            player_b_model = load_net(player_b[2], 874, PGNetwork, eval=True)
            models=[player_a_model, player_b_model]
            for i in range(1000):
                np.random.shuffle(deck_cards) 
                reward, record= play_game(1, Game, player_classes, 0, models, deck=deck_cards)
                player_a_rewards.append(reward[0][0])
                player_b_rewards.append(reward[1][0])
                records[i] = record[0]
            average_rewards[-1].append((sum(player_a_rewards)/len(player_a_rewards), sum(player_b_rewards)/len(player_b_rewards)))

            plt.hist(player_a_rewards, 
                label= player_a_label,
                color= player_a_color,
                alpha=0.6)
            plt.hist(player_b_rewards, 
                    label= player_b_label,
                    color= player_b_color,
                    alpha=0.6)
            plt.xlabel('score')
            plt.ylabel('count')
            
            plt.legend(loc='upper left')
            

            title = player_a[1] + "_" + player_b[1]
            plt.savefig(title + '.pdf') 
            plt.show()
            with open(title + ".json", "w") as outfile:
                json.dump(records, outfile) 
            # plt.savefig(title + '.pdf')  
            

    names = [player[1] for player in players]
    wins = pd.DataFrame(average_rewards, columns = names, index = names)
    wins = wins.transpose()
    wins.to_csv(r'average_rewards.csv')