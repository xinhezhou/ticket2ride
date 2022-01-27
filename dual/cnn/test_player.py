import numpy as np
import torch
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress
import random

import torch.nn as nn
import torch.nn.functional as F
from cnn.cnn_utils import generate_state_matrix


class TestPlayer:
    def __init__(self, num_colors, destination_cards, trains, id, model):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains = trains
        self.destination_cards = destination_cards
        self.id = id
        self.card_net = model["card_net"]
     

    def choose_route(self, game):
        """
        chooses a route based on the route policy network (route_net)
        and the availability of routes
        """
        availability = compute_availability_matrix(game.graph, game.status, self)
        available_routes = get_available_routes(availability)
        return random.choice(available_routes)
        

    def draw_or_claim(self, game):
        """
        If at least one route makes positive progress, claim a route (1). Otherwise, draw 2 cards (0)
        """
        graph = game.graph
        status = game.status
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        route_progress = []
        for route in available_routes:
            route_progress.append(compute_progress(graph, status, route, self.destination_cards, self.id))
        if max(route_progress) == 0:
            return 0
        else:
            return 1
   