import numpy as np
import json
import sys
sys.path.append("..")
from game import Game
from nonRLplayers.random_player import RandomPlayer
from nonRLplayers.greedy_player import GreedyPlayer
from nonRLplayers.frugal_player import FrugalPlayer
from RLplayers.dqn_player import DQNPlayer, QValueNetwork
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
    (1, 5),
    (2, 3),
    (0, 6)
]

def check_win(player, game):
    return len(player.destination_cards) == 0 or player.trains == 0 or game.card_index > len(game.cards)-3



def play_game(iterations, game_class, player_class, eps=0, model=None, update=False, deck = None):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    trains = []
    rounds = []
    losses = []
    rewards = []
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
        player = player_class(num_card_colors, destination_cards, 15, 1, model)
        game.draw_cards(player)
        game.draw_cards(player)
        record = {}
        record["deck"] = game.cards
        record["destinations"] = player.destination_cards
        actions = []
        reward = 0
        while not check_win(player, game):
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
                reward += compute_progress(game.graph, game.status, route, player.destination_cards, player.id) * coeff
                game.claim_route(route, player)
                actions.append(route)
            coeff *= decay

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




def compute_route_freq(records, k, win_file=None):
    route_freq = {}
    for key in records:
        record = records[key]
        actions = record["actions"]
        # print(actions[0])
        routes = []
        for i in range(0, len(actions)):
            action = actions[i]
            if len(action) == 3:
                routes.append(tuple(action[:2]))
        routes.sort()
        if tuple(routes) in route_freq:
            route_freq[tuple(routes)] += 1
        else:
            route_freq[tuple(routes)] = 1

    for key in route_freq:
        if route_freq[key] > k:
            print(key, route_freq[key])
    print(sorted(route_freq.values()))
    return route_freq




if __name__ == '__main__':
    average_rewards =[]
    rewards, records = play_game(1000, Game, RandomPlayer)
    with open("random_record_3.json", "w") as outfile:
        json.dump(records, outfile)
    print(sum(rewards)/len(rewards))



    rewards, records = play_game(1000, Game, GreedyPlayer)
    with open("greedy_record_3.json", "w") as outfile:
        json.dump(records, outfile)
    print(sum(rewards)/len(rewards))

    net_file = "dqn_3.pth.tar"
    model = load_net(net_file, 442, QValueNetwork, eval=True)
    rewards, records = play_game(1000, Game, DQNPlayer, eps=0, model=model)
    print(sum(rewards)/len(rewards))
    with open("dqn_eval_record_3.json", "w") as outfile:
        json.dump(records, outfile)



    # net_file = "es_checkpoints/60es_2.pth.tar"
    # model = load_net(net_file, 437, QValueNetwork, eval=True)
    # rewards, records = play_game(1000, Game, DQNPlayer, eps=0, model=model, deck=None)
    # print(sum(rewards)/len(rewards))
    # with open("es_eval_record_2.json", "w") as outfile:
    #     json.dump(records, outfile)


    # random_records = {}
    # random_rewards = []
    # greedy_records = {}
    # greedy_rewards = []
    # dqn_records = {}
    # dqn_rewards = []
    # es_records = {}
    # es_rewards = []
    # dqn_model = load_net(  "dqn_2.pth.tar", 437, QValueNetwork, eval=True)
    # es_model = load_net( "es_checkpoints/60es_2.pth.tar", 437, QValueNetwork, eval=True)
    # for i in range(1000):
    #     np.random.shuffle(deck_cards) 
    #     dqn_reward, dqn_record= play_game(1, Game, DQNPlayer, eps=0, model=dqn_model, deck=deck_cards)
    #     greedy_reward, greedy_record = play_game(1, Game, GreedyPlayer, deck=deck_cards)
    #     random_reward, random_record = play_game(1, Game, RandomPlayer, deck=deck_cards)
    #     es_reward, es_record = play_game(1, Game, DQNPlayer, eps=0, model=es_model, deck=deck_cards)

    #     random_rewards.append(random_reward[0])
    #     random_records[i] = random_record[0]
    #     greedy_rewards.append(greedy_reward[0])
    #     greedy_records[i] = greedy_record[0]
    #     dqn_rewards.append(dqn_reward[0])
    #     dqn_records[i] = dqn_record[0]
    #     es_rewards.append(es_reward[0])
    #     es_records[i] = es_record[0]
    
    # plt.hist(dqn_rewards, 
    #      label= "DQN",
    #      alpha=0.6)
    # plt.hist(es_rewards, 
    #         label='ES',
    #         alpha=0.6)
    # plt.hist(random_rewards, 
    #         label='Random', 
    #         alpha=0.6)
    # plt.hist(greedy_rewards, 
    #         label='Greedy', 
    #         alpha=0.6)

    # plt.xlabel('score')
    # plt.ylabel('count')
    
    # plt.legend(loc='upper left')
    # plt.title('Score Distributions')
    # plt.show()
        
    # with open("random_record.json", "w") as outfile:
    #     json.dump(random_records, outfile)
    # with open("greedy_record.json", "w") as outfile:
    #     json.dump(greedy_records, outfile)
    # with open("dqn_record.json", "w") as outfile:
    #     json.dump(dqn_records, outfile)
    # with open("es_record.json", "w") as outfile:
    #     json.dump(es_records, outfile) 


    # print(sum(random_rewards), sum(greedy_rewards), sum(dqn_rewards), sum(es_rewards))



    # f = open("random_record.json")
    # records = json.load(f)
    # route_freqs = compute_route_freq(records, 50, None)

    # f = open("greedy_record.json")
    # records = json.load(f)
    # route_freqs = compute_route_freq(records, 50, None)

    # f = open("dqn_record.json")
    # records = json.load(f)
    # route_freqs = compute_route_freq(records, 50, None)

    # f = open("es_record.json")
    # records = json.load(f)
    # route_freqs = compute_route_freq(records, 40, None)

