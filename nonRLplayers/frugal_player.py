import numpy as np
import random
from utils.game_utils import check_path, compute_availability_matrix, get_available_routes, compute_progress

class FrugalPlayer:
    def __init__(self, num_colors, destination_cards, trains, id, model=None):
        self.cards = num_colors * [0]
        self.routes = {}
        self.destination_cards = destination_cards
        self.trains = trains
        self.id = id


    def choose_route(self, game, players):
        """
        Find all possible routes and chooses a route that makes
        the most progress
        """
        graph = game.graph
        status = game.status
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        route_progress = []
        for route in available_routes:
            route_progress.append(compute_progress(graph, status, route, self.destination_cards, self.id))
        # print(route_progress)
        return available_routes[np.argmax(route_progress)]


    def draw_or_claim(self, game, players):
        """
        If there is at least one path that can be completed, claim a route (1).
        Otherwise, draw 2 cards (0)
        """
        graph = game.graph
        status = game.status
        availability = compute_availability_matrix(graph, status, self)
        for a, b in self.destination_cards:
            if  check_path(availability, a, b):
                return 1
        return 0
    
    def duplicate(self):
        copy = FrugalPlayer(len(self.cards), self.destination_cards[:], self.trains, self.id)
        copy.cards = self.cards[:]
        return copy

