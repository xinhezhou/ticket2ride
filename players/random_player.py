import numpy as np
import random
from utils.game_utils import get_available_routes, compute_availability_matrix

class RandomPlayer:
    def __init__(self, num_colors, destination_cards, model=None, id=1):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0
        self.destination_cards = destination_cards
        self.id = id


    def choose_route(self, graph, status):
        """
        Find all possible routes and randomly choose one to take
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        return random.choice(available_routes)

    def draw_or_claim(self, graph, status):
        """
        Randomly decide whether to draw 2 more cards (0) or claim a route (1)
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0 or random.random() < 0.4:
            return 0
        else:
            return 1