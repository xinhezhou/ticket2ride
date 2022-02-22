import numpy as np
import random
from utils.game_utils import get_available_routes, compute_availability_matrix

class RandomPlayer:
    def __init__(self, num_colors, destination_cards, trains, id):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains = trains
        self.destination_cards = destination_cards
        self.id = id


    def choose_route(self, game, players):
        """
        Find all possible routes and randomly choose one to take
        """
        availability = compute_availability_matrix(game.graph, game.status, self)
        available_routes = get_available_routes(availability)
        return random.choice(available_routes)

    def draw_or_claim(self, game, players):
        """
        Randomly decide whether to draw 2 more cards (0) or claim a route (1)
        """
        if random.random() < 0.3:
            return 0
        else:
            return 1

    def duplicate(self):
        copy = RandomPlayer(len(self.cards), self.destination_cards[:], self.trains, self.id)
        copy.cards = self.cards[:]
        return copy