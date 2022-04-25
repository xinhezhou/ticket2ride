

import json
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import sys
import numpy as np
sys.path.append("..")
from nonRLplayers.greedy_player import GreedyPlayer
from RLplayers.exprience_replay import ReplayMemory
from RLplayers.rl_utils import push_memory, load_net
from RLplayers.pg_player import PGPlayer, PGNetwork, compute_input_matrix
from RLplayers.exprience_replay import ReplayMemory, Transition
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress
from game import Game

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
    [(0, 6), (3, 6)]
]


def check_win(players, game):
    for player in players:
        if len(player.destination_cards) == 0 or player.trains == 0:
            return True
    return game.card_index > len(game.cards)-3

def discounted_returns(rewards, gamma):
    returns = torch.zeros_like(rewards)
    #### Your code here
    returns[-1] = rewards[-1]
    for i, r in enumerate(reversed(rewards[:-1])):
      returns[-i-2] = r + gamma * returns[-i-1]
    
    return returns

def optimize_model(policy_net, optimizer, rewards, log_probs,gamma):
    # compute policy losses
    returns = discounted_returns(rewards, gamma)
    eps = np.finfo(np.float32).eps.item()
    discount = torch.zeros_like(rewards)
    discount[0] = 1
    for i in range(1, len(discount)):
        discount[i] = gamma * discount[i-1]


    # loss = - torch.sum(log_probs) * returns[0]
    loss = - torch.sum(log_probs * returns * discount)
    # parameter update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def rollout_attacker(models, game_class, player_classes, deck=None, MAX_T=100, gamma=0.9):
    actions = torch.zeros(MAX_T, dtype=torch.int)
    rewards = torch.zeros(MAX_T,)
    log_probs = torch.zeros(MAX_T, )
    values = torch.zeros(MAX_T)
    T = 0
    ep_reward = 0
    deck_cards = [0] * 12  + [1,2,3,4,5,6] * 10 # train cards in the deck
    if deck is None:
        np.random.shuffle(deck_cards) 
    else:
        deck_cards = deck
    game = game_class(num_vertices, num_route_colors, edges, deck_cards)
    players = []
    for i in range(len(player_classes)):
        player = player_classes[i](num_card_colors, destination_cards[i], 10, i+1, models[i])
        game.draw_cards(player)
        game.draw_cards(player)
        players.append(player)
    reward = 0
    decay = 1

    while T < MAX_T:
        pg_player = players[0]
        availability = compute_availability_matrix(game.graph, game.status, pg_player)
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0:
            mask = torch.cat([torch.tensor([[1]]), torch.zeros(7*7*7, 1)])
        else:
            mask = torch.cat([torch.tensor([[1]]), torch.reshape(torch.from_numpy(availability), (7*7*7, 1))])
        dist = models[0](compute_input_matrix(game, players),mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if action == 0:
            cards = game.draw_cards(pg_player)
            reward = 0
        else:
            action_ = action.item() - 1
            u = action_ // 49
            v = (action_ - 49 * u) // 7
            c = action_ - 49 * u - 7 * v
            route = u,v,c
            reward = compute_progress(game.graph, game.status, route, pg_player.destination_cards, pg_player.id)
            game.claim_route(route, pg_player)

        
        nonrl_player = players[1]
        availability = compute_availability_matrix(game.graph, game.status, nonrl_player)
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0 or nonrl_player.draw_or_claim(game, players) == 0:
            cards = game.draw_cards(nonrl_player)
        else:
            route = nonrl_player.choose_route(game, players)
            # reward -= compute_progress(game.graph, game.status, route, greedy_player.destination_cards, greedy_player.id)
            game.claim_route(route, nonrl_player)


        # print(ep_reward, reward, decay)
        ep_reward += reward * decay
        rewards[T] = reward
        actions[T] = action
        log_probs[T] = log_prob
        decay *= gamma
        T += 1
        
        if check_win(players, game):
            break
    return actions[:T], rewards[:T], log_probs[:T], values[:T], ep_reward


def rollout_defender(models, game_class, player_classes, deck=None, MAX_T=100, gamma=0.9):
    actions = torch.zeros(MAX_T, dtype=torch.int)
    rewards = torch.zeros(MAX_T,)
    log_probs = torch.zeros(MAX_T, )
    values = torch.zeros(MAX_T)
    T = 0
    ep_reward = 0
    deck_cards = [0] * 12  + [1,2,3,4,5,6] * 10 # train cards in the deck
    if deck is None:
        np.random.shuffle(deck_cards) 
    else:
        deck_cards = deck
    game = game_class(num_vertices, num_route_colors, edges, deck_cards)
    players = []
    for i in range(len(player_classes)):
        player = player_classes[i](num_card_colors, destination_cards[i], 10, i+1, models[i])
        game.draw_cards(player)
        game.draw_cards(player)
        players.append(player)
    reward = 0
    decay = 1

    while T < MAX_T:
        nonrl_player = players[0]
        availability = compute_availability_matrix(game.graph, game.status, nonrl_player)
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0 or nonrl_player.draw_or_claim(game, players) == 0:
            cards = game.draw_cards(nonrl_player)
        else:
            route = nonrl_player.choose_route(game, players)
            # reward -= compute_progress(game.graph, game.status, route, greedy_player.destination_cards, greedy_player.id)
            game.claim_route(route, nonrl_player)

        pg_player = players[1]
        availability = compute_availability_matrix(game.graph, game.status, pg_player)
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0:
            mask = torch.cat([torch.tensor([[1]]), torch.zeros(7*7*7, 1)])
        else:
            mask = torch.cat([torch.tensor([[1]]), torch.reshape(torch.from_numpy(availability), (7*7*7, 1))])
        dist = models[1](compute_input_matrix(game, players),mask)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if action == 0:
            cards = game.draw_cards(pg_player)
            reward = 0
        else:
            # print("here")
            action_ = action.item() - 1
            u = action_ // 49
            v = (action_ - 49 * u) // 7
            c = action_ - 49 * u - 7 * v
            route = u,v,c
            reward = compute_progress(game.graph, game.status, route, pg_player.destination_cards, pg_player.id)
            game.claim_route(route, pg_player)

        
        
        ep_reward += reward * decay
        rewards[T] = reward
        actions[T] = action
        log_probs[T] = log_prob
        decay *= gamma
        T += 1
        
        if check_win(players, game):
            break
    return actions[:T], rewards[:T], log_probs[:T], values[:T], ep_reward

if __name__ == '__main__':
    # # """
    # # TEST 1: PG vs greedy 
    # # """
    # initial_checkpoint = None
    # selfplay_checkpoint ="pg_greedy.pth.tar"
    # reward_file = "pg_greedy_reward.pdf"
    # policy_net = load_net(initial_checkpoint, 874, PGNetwork)
    # optimizer = optim.Adam(policy_net.parameters(),  lr=0.0001)
    # gamma = 0.9
    # # torch.autograd.set_detect_anomaly(True)

    # running_reward = 0
    # history_reward = [] # store moving average of empirical rewards
    # losses = []
    # for step in range(10000):
    #     actions, rewards, log_probs, values, ep_reward = rollout_attacker([policy_net, None], Game, [PGPlayer, GreedyPlayer], deck=[0] * 12  + [1,2,3,4,5,6] * 10 )
    #     running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    #     history_reward.append(running_reward)
    #     loss = optimize_model(policy_net, optimizer, rewards, log_probs, gamma)
    #     losses.append(loss)
    #     # print(loss)
    #     if step % 100 == 0:
    #         # print(actions, rewards)
    #         print('Episode {}\tLast reward: {:.2f} \tAverage reward: {:.2f}'.format(
    #               step, ep_reward, running_reward))
    #         # Saves model checkpoint
    #         torch.save({
    #             'state_dict': policy_net.state_dict(),
    #         }, selfplay_checkpoint)
    #         plt.clf()
    #         plt.plot(range(len(history_reward)), history_reward)
    #         plt.savefig(reward_file)

    
    # """
    # TEST 2: greedy vs pg
    # """
    # memory = ReplayMemory(100000)
    # initial_checkpoint = None
    # selfplay_checkpoint ="greedy_pg.pth.tar"
    # reward_file = "greedy_pg_reward.pdf"
    # policy_net = load_net(initial_checkpoint, 874, PGNetwork)
    # optimizer = optim.Adam(policy_net.parameters(),  lr=0.0001)
    # gamma = 0.9
    # # torch.autograd.set_detect_anomaly(True)



    # running_reward = 0
    # history_reward = [] # store moving average of empirical rewards
    # losses = []
    # for step in range(10000):
    #     actions, rewards, log_probs, values, ep_reward = rollout_defender([None,policy_net], Game, [GreedyPlayer, PGPlayer], deck=[0] * 12  + [1,2,3,4,5,6] * 10 )
    #     running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
    #     history_reward.append(running_reward)
    #     loss = optimize_model(policy_net, optimizer, rewards, log_probs, gamma)
    #     losses.append(loss)
    #     # print(loss)
    #     if step % 100 == 0:
    #         # print(actions, rewards)
    #         print('Episode {}\tLast reward: {:.2f} \tAverage reward: {:.2f}'.format(
    #               step, ep_reward, running_reward))
    #         # Saves model checkpoint
    #         torch.save({
    #             'state_dict': policy_net.state_dict(),
    #         }, selfplay_checkpoint)
    #         plt.clf()
    #         plt.plot(range(len(history_reward)), history_reward)
    #         plt.savefig(reward_file)

    # """
    # TEST 3: PG vs PG
    # """
    initial_defender = None
    # initial_attacker = None
    defender_checkpoint ="greed_pg.pth.tar"
    defender_reward_file = "greedy_pg_reward.pdf"
    # attacker_checkpoint ="pg_greedy.pth.tar"
    # attacker_reward_file = "pg_greedy.pdf"
    # attacker_net = load_net(initial_attacker, 874, PGNetwork)
    defender_net = load_net(initial_defender, 874, PGNetwork)
    # attacker_optimizer = optim.Adam(attacker_net.parameters(),  lr=0.0001)
    defender_optimizer = optim.Adam(defender_net.parameters(),  lr=0.0001)
    gamma = 0.9
    # torch.autograd.set_detect_anomaly(True)

    # attacker_running_reward = 0
    defender_running_reward = 0
    # attacker_history_reward = [] # store moving average of empirical rewards
    defender_history_reward = []
    for step in range(20000):
        # actions, rewards, log_probs, values, attacker_ep_reward = rollout_attacker([attacker_net, None], Game, [PGPlayer, GreedyPlayer], deck=None)
        # attacker_running_reward = 0.05 * attacker_ep_reward + (1 - 0.05) * attacker_running_reward
        # attacker_history_reward.append(attacker_running_reward)
        # loss = optimize_model(attacker_net, attacker_optimizer, rewards, log_probs, gamma)

        actions, rewards, log_probs, values, defender_ep_reward = rollout_defender([None, defender_net], Game, [GreedyPlayer, PGPlayer], deck=None)
        defender_running_reward = 0.05 * defender_ep_reward + (1 - 0.05) * defender_running_reward
        defender_history_reward.append(defender_running_reward)
        loss = optimize_model(defender_net, defender_optimizer, rewards, log_probs, gamma)

        # print(loss)
        if step % 100 == 0:
            # print(actions, rewards)
            # print('Episode {}\t Attacker Last reward: {:.2f} \tAttacker Average reward: {:.2f}'.format(
            #       step, attacker_ep_reward, attacker_running_reward))
            # Saves model checkpoint
            # torch.save({
            #     'state_dict': attacker_net.state_dict(),
            # }, attacker_checkpoint)
            # plt.clf()
            # plt.plot(range(len(attacker_history_reward)), attacker_history_reward)
            # plt.savefig(attacker_reward_file)

            # print(actions, rewards)
            print('Episode {}\t Defender Last reward: {:.2f} \tDefender Average reward: {:.2f}'.format(
                  step, defender_ep_reward, defender_running_reward))
            # Saves model checkpoint
            torch.save({
                'state_dict': defender_net.state_dict(),
            }, defender_checkpoint)
            plt.clf()
            plt.plot(range(len(defender_history_reward)), defender_history_reward)
            plt.savefig(defender_reward_file)