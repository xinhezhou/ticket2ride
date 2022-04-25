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



if __name__ == '__main__':
    # rewards, records = play_game(1, Game, [RandomPlayer, GreedyPlayer], models=[None, None], deck=[0] * 12  + [1,2,3,4,5,6] * 10)
    # print(sum(rewards)/len(rewards))
    # rewards, records = play_game(1000, Game, [GreedyPlayer, GreedyPlayer], models=[None, None], deck=None)
    # print(sum(rewards[0]), sum(rewards[1]))
   
    # attacker_net = load_net("pg/pg_greedy.pth.tar", 874, PGNetwork)
    # rewards, records = play_game(1000, Game, [PGPlayer, GreedyPlayer], models=[attacker_net, None], deck=None)
    # print(sum(rewards[0]), sum(rewards[1]))
    # with open("es_random_record.json", "w") as outfile:
    #     json.dump(records, outfile) 

    # defender_net = load_net("pg/greedy_pg.pth.tar", 874, PGNetwork)
    # rewards, records = play_game(1000, Game, [GreedyPlayer, PGPlayer], models=[None,defender_net], deck=None)
    # print(sum(rewards[0]), sum(rewards[1]))
    # with open("greedy_pg_record.json", "w") as outfile:
    #     json.dump(records, outfile) 

    # f = open("pg_random_record.json")
    # records = json.load(f)
    # route_freqs = compute_route_freq(records, 40, None)

    # f = open("greedy_pg_record.json")
    # records = json.load(f)
    # route_freqs = compute_route_freq(records, 25, None)

    random_records = {}
    random_rewards = []
    greedy_records = {}
    greedy_rewards = []
    pg_records = {}
    pg_rewards = []
    es_records = {}
    es_rewards = []
    pg_model = load_net( "pg/pg_greedy.pth.tar", 874, PGNetwork, eval=True)
    es_model = load_net( "pg/es_greedy.pth.tar", 874, PGNetwork, eval=True)
    for i in range(1000):
        np.random.shuffle(deck_cards) 
        pg_reward, pg_record= play_game(1, Game, [PGPlayer, RandomPlayer], models=[pg_model, None], deck=deck_cards)
        greedy_reward, greedy_record = play_game(1, Game, [GreedyPlayer, RandomPlayer], models=[pg_model, None], deck=deck_cards)
        random_reward, random_record = play_game(1, Game, [RandomPlayer, RandomPlayer], models=[pg_model, None], deck=deck_cards)
        es_reward, es_record = play_game(1, Game, [PGPlayer, RandomPlayer], models=[es_model, None], deck=deck_cards)

        random_rewards.append(random_reward[0][0])
        random_records[i] = random_record[0]
        greedy_rewards.append(greedy_reward[0][0])
        greedy_records[i] = greedy_record[0]
        pg_rewards.append(pg_reward[0][0])
        pg_records[i] = pg_record[0]
        es_rewards.append(es_reward[0][0])
        es_records[i] = es_record[0]

    plt.hist(pg_rewards, 
         label= "Policy Gradient",
         alpha=0.6)
    plt.hist(es_rewards, 
            label='ES',
            alpha=0.6)
    plt.hist(random_rewards, 
            label='Random', 
            alpha=0.6)
    plt.hist(greedy_rewards, 
            label='Greedy', 
            alpha=0.6)

    plt.xlabel('score')
    plt.ylabel('count')
    
    plt.legend(loc='upper left')
    plt.title('Score Distributions')
    plt.show()
    