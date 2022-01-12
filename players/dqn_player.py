import numpy as np
import torch
from utils.dqn_utils import compute_dqn_input
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress
import random

import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, device, learning_rate=1e-4):
        super().__init__()
        self.num_actions = num_actions
        self.device = device
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

        
    def forward(self, x):
        # print(x)
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=0)
        return x.view(x.size(0), -1)


class DQNPlayer:
    def __init__(self, num_colors, destination_cards, model, id=1):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0 
        self.destination_cards = destination_cards
        self.id = id
            
        self.card_net = model["card_net"]
        self.route_net = model["route_net"]
        self.card_optimizer = model["card_optimizer"]
        self.route_optimizer = model["route_optimizer"]
        self.loss_fn = model["loss_fn"]
        self.gamma = model["gamma"]


    def choose_route(self, graph, status):
        """
        chooses a route based on the route policy network (route_net)
        and the availability of routes
        """
        availability = compute_availability_matrix(graph, status, self)
        if random.random() < 0.2:
            available_routes = get_available_routes(availability)
            return random.choice(available_routes)
        dqn_input = compute_dqn_input(graph, status, self.cards)
        dqn_output = self.route_net(dqn_input)
        dqn_output = torch.reshape(dqn_output, availability.shape)
        action_dist = torch.mul(dqn_output, torch.from_numpy(availability))
        
        # compute the availabile route with the highest value
        best_route = 0,0,0
        best_action_value = 0
        for u in range(len(availability)):
            for v in range(len(availability)):
                for c in range(len(availability[0][0])):
                    if action_dist[u][v][c] > best_action_value:
                        best_action_value = action_dist[u][v][c]
                        best_route = u,v,c
        # print(best_route)
        return best_route
        

    def draw_or_claim(self, graph, status):
        """
        Choose the action associated withe the highest value
        based on the card policy
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0:
            return 0
        elif random.random() < 0.2:
            return 1
        dqn_input = compute_dqn_input(graph, status, self.cards)
        dqn_output = self.card_net(dqn_input)
        # print(dqn_output)
        return torch.argmax(dqn_output)

    def update_model(self, graph, current_status, next_status, current_cards, next_cards, route):
        """
        Update policy networks through back propagation
        """
        u,v,c = route
        reward = compute_progress(graph, current_status, route, self.destination_cards)
        current_input = compute_dqn_input(graph, current_status, current_cards)
        next_input = compute_dqn_input(graph, next_status, next_cards)

        # update the card policy net
        current_q_values = self.card_net(current_input)
        current_q_value = current_q_values[1]
        next_best_q = max(self.card_net(next_input))
        expected_q_avlue = reward + self.gamma * next_best_q
        loss = self.loss_fn(current_q_value, expected_q_avlue)
        self.card_optimizer.zero_grad()
        loss.backward()
        self.card_optimizer.step()

        # update the route policy net
        current_q_values = self.route_net(current_input)
        current_q_value = torch.reshape(current_q_values, graph.shape)[u][v][c]
        next_best_q = max(self.route_net(next_input))
        expected_q_avlue = reward + self.gamma * next_best_q
        loss = self.loss_fn(current_q_value, expected_q_avlue)
        self.route_optimizer.zero_grad()
        loss.backward()
        self.route_optimizer.step()

        return reward, loss.item()