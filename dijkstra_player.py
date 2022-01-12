import numpy as np
import random
from game_utils import compute_availability_matrix, get_available_routes, compute_progress

class DijkstraPlayer:
    def __init__(self, num_colors, destination_cards, model=None):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0 
        self.explored = {}
        self.destination_cards = destination_cards


    def choose_route(self, graph, status):
        """
        Find all possible routes and chooses a route that connects to a city that
        has made the most progress
        """
        # print(self.explored)
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        route_progress = []
        for route in available_routes:
            route_progress.append(compute_progress(graph, status, route, self.destination_cards))
        # print(route_progress)
        return available_routes[np.argmax(route_progress)]

    def draw_or_claim(self, graph, status):
        """
        If at least one route connects to an explored city, claim a route (1). Otherwise, draw 2 cards (0)
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

    def update_model(self, graph, current_status, next_status, current_cards, next_cards, route):
        pass