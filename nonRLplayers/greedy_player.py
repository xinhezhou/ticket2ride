import numpy as np
import random
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress

class GreedyPlayer:
    def __init__(self, num_colors, destination_cards, trains, id):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains = trains
        self.destination_cards = destination_cards
        self.trains = trains
        self.id = id


    def choose_route(self, game, players):
        """
        Find all possible routes and chooses a route that makes the most progress. 
        Here, progress is defined as the decrease in the sum of distances
        between cities on the destination cards
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
    
    def duplicate(self):
        copy = GreedyPlayer(len(self.cards), self.destination_cards[:], self.trains, self.id)
        copy.cards = self.cards[:]
        return copy
