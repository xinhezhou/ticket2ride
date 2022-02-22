import numpy as np
import sys 
sys.path.append("..")
from utils.game_utils import check_path

class Game:
    def __init__(self, num_vertices, num_colors, edges, deck_cards):
        self.num_vertices = num_vertices
        self.num_colors = num_colors
        self.edges = edges
        self.graph = np.zeros((num_vertices, num_vertices, num_colors)) # (u, v, c) -> w
        self.status = np.zeros((num_vertices, num_vertices, num_colors)) # (u, v, c) -> -1/0/1 (not available/free/occupied player id)
        for u in range(num_vertices):
            for v in range(num_vertices):
                for c in range(num_colors):
                    if (u, v, c) in edges:
                        self.graph[u][v][c] = edges[(u, v, c)]
                    else:
                        self.status[u][v][c] = -1
        
        self.cards = deck_cards
        self.card_index = 0


        
    def claim_route(self, route, player):
        """
        claim a route using train cards on a player's hand 
        route: must be valid and available
        The following variables will be upated:
            - game status
            - number of trains used by player
            - routes claimed by the player
            - train cards owned by the player 
            - destination cards owned by the player
        """
        u,v,c = route
        assert u < v
        self.status[u][v][c] = player.id
        
        count = self.graph[u][v][c]
        player.trains -= count
        player.routes[(u,v)] = c
        if player.cards[c] >= count:
            player.cards[c] -= count
        else:
            player.cards[0] -= count - player.cards[c]
            
        incomplete_destinations = []
        for u, v in player.destination_cards:
            if not check_path(self.status, u, v, player.id):
                incomplete_destinations.append((u,v))
        player.destination_cards = incomplete_destinations


    def draw_cards(self, player):
        """
        draw 2 cards from the private deck
        """
        cards = self.cards[self.card_index:self.card_index+2]
        player.cards[self.cards[self.card_index]] += 1
        player.cards[self.cards[self.card_index+1]] += 1
        self.card_index += 2
        return cards

    def duplicate(self):
        copy = Game(self.num_vertices, self.num_colors, self.edges, self.cards)
        copy.graph = np.copy(self.graph)
        copy.status = np.copy(self.status)
        copy.card_index = self.card_index
        return copy





