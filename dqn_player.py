import numpy as np
import torch
from dqn_utils import compute_card_agent_input
from game_utils import compute_availability_matrix

class Player:
    def __init__(self, num_trains, model):
        self.cards = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.trains = num_trains
        self.model = model

    def choose_route(self, graph, availability):
        if np.random.random_sample() < 0.2:
            return None

        available_routes = []
        v = len(availability)
        c = len(availability[0][0])
        for i in range(v):
            for j in range(v):
                for k in range(c):
                    if availability[i][j][k] and self.trains > graph[i][j][k]:
                        available_routes.append((i,j,k))

        if len(available_routes) == 0:
            return None
        else:
            return available_routes[np.random.randint(len(available_routes))]

    def choose_card(self, graph, status, public_cards):
        availability = compute_availability_matrix(graph, status, self.cards)
        action_dist = self.model(compute_card_agent_input(graph, availability, public_cards, self.cards))
        action = torch.argmax(action_dist)
        available_colors = []
        for i in range(9):
            if public_cards[i] > 0:
                available_colors.append(i)
    
        if action_dist[action] == 0 or public_cards[action] == 0:
            return None, np.random.choice(available_colors)
        else:
            return action_dist, action