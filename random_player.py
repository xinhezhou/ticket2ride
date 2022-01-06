import numpy as np
import random
from game_utils import get_available_routes

class RandomPlayer:
    def __init__(self, num_colors, start, end, model=None):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0 
        self.explored = {start: 0,  end: 1}


    def choose_route(self, graph, availability):
        """
        Find all possible routes and randomly choose one to take
        """
        available_routes = get_available_routes(availability)
        if len(available_routes) == 0:
            return None
        else:
            return random.choice(available_routes)

    def draw_or_claim(self, graph, availability):
        """
        Randomly decide whether to draw 2 more cards or claim a route
        """
        if random.random() < 0.5:
            return 0
        else:
            return 1