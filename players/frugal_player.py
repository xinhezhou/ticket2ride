import numpy as np
import random
from utils.game_utils import compute_availability_matrix, get_available_routes, compute_progress

class FrugalPlayer:
    def __init__(self, num_colors, destination_cards, model=None, id=1):
        self.cards = num_colors * [0]
        self.routes = {}
        self.destination_cards = destination_cards
        self.trains_used = 0
        self.id = id


    def choose_route(self, graph, status):
        """
        Find all possible routes and chooses a route that makes
        positive progress and uses the fewest wild cards and the 
        smallest fraction of existing cards of that color
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        route_progress = []
        for route in available_routes:
            route_progress.append(compute_progress(graph, status, route, self.destination_cards))
        card_utility = []
        for i in range(len(available_routes)):
            u,v,c = available_routes[i]
            if route_progress[i] == 0:
                card_utility.append(100)
            elif self.cards[c] >= graph[u][v][c]:
                card_utility.append(graph[u][v][c] / self.cards[c] - 0.0001)
            else:
                card_utility.append(graph[u][v][c] - self.cards[c])
        # print(available_routes[np.argmin(card_utility)])
        return available_routes[np.argmin(card_utility)]

    def draw_or_claim(self, graph, status):
        """
        If there is at least one route, claim a route (1).
        Otherwise, draw 2 cards (0)
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
