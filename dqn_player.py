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

        available_routes = []
        route_lengths = []
        v = len(availability)
        c = len(availability[0][0])
        for i in range(v):
            for j in range(v):
                for k in range(c):
                    if availability[i][j][k] and self.trains >= graph[i][j][k]:
                        available_routes.append((i,j,k))
                        route_lengths.append(graph[i][j][k])

        if len(available_routes) == 0:
            return None
        else:
            if (3, 5, 9) in available_routes:
                return (3, 5, 9)
            else:
                return available_routes[np.argmax(route_lengths)]
                

    def choose_card(self, graph, status, public_cards):
        availability = compute_availability_matrix(graph, status, self.cards)
        action_dist = self.model(compute_card_agent_input(graph, availability, public_cards, self.cards))
        action_dist_available = []
        for i in range(9):
            if public_cards[i] == 0:
                action_dist_available.append(0)
            else:
                action_dist_available.append(action_dist[i])
    
        return action_dist, np.argmax(action_dist_available)