import numpy as np
import random
from game_utils import get_available_routes

class GreedyPlayer:
    def __init__(self, num_colors, model=None):
        self.cards = num_colors * [0]
        self.routes = {}
        self.explored = set()
        self.trains_used = 0 


    def choose_route(self, graph, availability):
        """
        Find all possible routes and chooses a route that connects to a city that
        has been explored.
        Ties are first broken by the number of explored cities it connects to, then
        by length of route (the shorter the better), and then by alphabetical order
        of the origin    
        """
        available_routes = get_available_routes(availability)
        one_explored = []
        two_explored = []
        for u, v, c in available_routes:
            if u in self.explored and v in self.explored:
                two_explored.add((graph[u][v][c], u, v, c))
            elif u in self.explored or v in self.explored:
                one_explored.add((graph[u][v][c], u, v, c))
        one_explored.sort()
        two_explored.sort()

        if len(two_explored) > 0:
            return two_explored[0][1:]
        elif len(two_explored) > 0:
            return one_explored[0][1:]
        else:
            return None

    def draw_or_claim(self, availability):
        """
        If at least one route connects to an explored city, claim a route. Otherwise, draw 2 cards
        """
        available_routes = get_available_routes(availability)
        for u, v, c in available_routes:
            if u in self.explored_cities or v in self.explored_cities:
                return 0
        return 1