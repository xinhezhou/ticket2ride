import numpy as np

class Game:
    def __init__(self, num_vertices, num_colors, edges, deck_cards):
        self.graph = np.zeros((num_vertices, num_vertices, num_colors)) # (u, v, c) -> w
        self.status = np.zeros((num_vertices, num_vertices, num_colors)) # (u, v, c) -> -1/0/1 (not available/free/occupied)

        self.cards = deck_cards
        self.card_index = 0

        for u in range(num_vertices):
            for v in range(num_vertices):
                for c in range(num_colors):
                    if (u, v, c) in edges:
                        self.graph[u][v][c] = edges[(u, v, c)]

        

    def claim_route(self, route, player):
        """
        claim a route using train cards on a player's hand 
        route: must be valid and available
        player.cards and player.trains will be modified after train cards are used
        """
        u,v,c = route
        # print(u, v, c)
        assert u < v
        self.status[u][v][c] = 1
        count = self.graph[u][v][c]
        player.trains_used += count
        player.routes[(u,v)] = c
        player.explored[u] = len(player.explored)
        player.explored[v] = len(player.explored)

        if player.cards[c] >= count:
            player.cards[c] -= count
        else:
            player.cards[0] -= count - player.cards[c]
            player.cards[c] = 1

    def draw_cards(self, player):
        """
        draw 2 cards from the private deck
        """
        player.cards[self.cards[self.card_index]] += 1
        player.cards[self.cards[self.card_index+1]] += 1
        self.card_index += 2



