import numpy as np
import torch
from dqn_utils import compute_dqn_input
from game_utils import compute_availability_matrix, get_available_routes, compute_progress

class DQNPlayer:
    def __init__(self, num_colors, start, end, model):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0 
        self.start = start
        self.end = end
        self.explored = {start: 0,  end: 1}
        self.net = model["net"]
        self.loss_fn = model["loss_fn"]
        self.optimizer = model["optimizer"]
        self.gamma = model["gamma"]


    def choose_route(self, graph, status):
        """
        chooses a route based on the policy network (model)
        """
        availability = compute_availability_matrix(graph, status, self)
        dqn_input = compute_dqn_input(graph, status, self.cards)
        dqn_output = self.net(dqn_input)
        # print(dqn_input.shape, dqn_output.shape, availability.shape)
        dqn_output = torch.reshape(dqn_output, availability.shape)
        action_dist = torch.mul(dqn_output, torch.from_numpy(availability))
        best_route = 0,0,0
        best_action_value = 0
        for u in range(len(availability)):
            for v in range(len(availability)):
                for c in range(len(availability[0][0])):
                    if action_dist[u][v][c] > best_action_value:
                        best_action_value = action_dist[u][v][c]
                        best_route = u,v,c
        return best_route
        
        

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

    def update_model(self, graph, current_status, next_status, current_cards, next_cards, route):
        u,v,c = route
        reward = compute_progress(graph, current_status, route, self.start, self.end)
        current_q_values = self.net(compute_dqn_input(graph, current_status, current_cards))
        current_q_value = torch.reshape(current_q_values, graph.shape)[u][v][c]
        next_input = compute_dqn_input(graph, next_status, next_cards)
        next_best_q = max(self.net(next_input))
        expected_q_avlue = reward + self.gamma * next_best_q

        loss = self.loss_fn(current_q_value, expected_q_avlue)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return reward, loss.item()