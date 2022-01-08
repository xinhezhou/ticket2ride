import numpy as np
import torch
from dqn_utils import compute_dqn_input
from game_utils import compute_availability_matrix, get_available_routes, compute_progress

class Player:
    def __init__(self, num_colors, start, end, model):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0 
        self.explored = {start: 0,  end: 1}

    def choose_route(self, graph, status):
        """
        chooses a route based on the policy network (model)
        """
        availability = compute_availability_matrix(graph, status, self)
        dqn_input = compute_dqn_input(graph, status, self)
        dqn_output = np.reshape(self.model(dqn_input), availability.shape)
        action_dist = np.dot(dqn_output, availability)
        return np.argmax(action_dist)
        
        

    def draw_or_claim(self, graph, status):
        """
        If at least one route connects to an explored city, claim a route. Otherwise, draw 2 cards
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        for u, v, c in available_routes:
            if u in self.explored or v in self.explored:
                return 1
        return 0

    def update_model(self, graph, prev_status, next_status, prev_cards, next_cards, route):
        reward = compute_progress(graph, prev_status, route, self.start, self.end)
        prev_q_values = compute_dqn_input(graph, next_status, prev_cards)
        next_input = compute_dqn_input(graph, next_status, next_cards)
        next_best_q = max(self.model(next_input))
        u,v,c = route
        current_q_value = prev_q_values.reshape(graph.shape)[u][v][c]
        expected_q_avlue = reward + self.gamma * next_best_q

        loss = self.criterion(current_q_value, expected_q_avlue)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()