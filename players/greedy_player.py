import numpy as np
import random
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress

class GreedyPlayer:
    def __init__(self, num_colors, destination_cards, model=None, id=1):
        self.cards = num_colors * [0]
        self.routes = {}
        self.destination_cards = destination_cards
        self.trains_used = 0
        self.id = id


    def choose_route(self, graph, status):
        """
        Find all possible routes and chooses a route that makes the most progress. 
        Here, progress is defined as the decrease in the sum of distances
        between cities on the destination cards
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        route_progress = []
        for route in available_routes:
            route_progress.append(compute_progress(graph, status, route, self.destination_cards))
        # print(route_progress)
        return available_routes[np.argmax(route_progress)]

    def draw_or_claim(self, graph, status):
        """
        If at least one route makes positive progress, claim a route (1). Otherwise, draw 2 cards (0)
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        route_progress = []
        for route in available_routes:
            route_progress.append(compute_progress(graph, status, route, self.destination_cards))
        if len(route_progress) == 0 or max(route_progress) == 0:
            return 0
        else:
            return 1
