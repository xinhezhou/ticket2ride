import numpy as np
import random
from game_utils import get_available_routes, compute_availability_matrix

class GreedyPlayer:
    def __init__(self, num_colors, destination_cards, model=None):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains_used = 0 
        self.explored = {}
        for u,v in destination_cards:
            self.explored[u] = len(self.explored)
            self.explored[v] = len(self.explored)
        self.destination_cards = destination_cards


    def choose_route(self, graph, status):
        """
        Find all possible routes and chooses a route that connects to a city that
        has been explored recently.
        Ties are first broken by the number of explored cities it connects to, then
        by how recent the exploration is
        """
        # print(self.explored)
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        one_explored = []
        two_explored = []
        for u, v, c in available_routes:
            if u in self.explored and v in self.explored:
                two_explored.append((graph[u][v][c], u, v, c))
            elif u in self.explored:
                one_explored.append((self.explored[u], u, v, c))
            elif v in self.explored:
                one_explored.append((self.explored[v], u, v, c))
        one_explored.sort()
        two_explored.sort()

        if len(two_explored) > 0:
            return two_explored[-1][1:]
        elif len(one_explored) > 0:
            return one_explored[-1][1:]
        else:
            return None

    def draw_or_claim(self, graph, status):
        """
        If at least one route connects to an explored city, claim a route. Otherwise, draw 2 cards
        """
        availability = compute_availability_matrix(graph, status, self)
        available_routes = get_available_routes(availability)
        if len(available_routes) > 0:
            return 1
        else:
            return 0
