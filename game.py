import numpy as np

class Game:
    def __init__(self, num_vertices, num_colors, edges):
        self.v = num_vertices
        self.c = num_colors
        self.graph = np.zeros((num_vertices, num_vertices, num_colors)) # (u, v, c) -> w
        self.status = np.zeros((num_vertices, num_vertices, num_colors)) # (u, v, c) -> -1/0/1 (non-existent/free/occupied)

        self.deck_cards = [0] * 14  + [1,2,3,4,5,6,7,8] * 12 # train cards in the deck
        np.random.shuffle(self.deck_cards) 
        self.public_cards = [0, 0, 0, 0, 0, 0, 0, 0, 0] 
        for i in range(5):
            self.public_cards[self.deck_cards[i]] += 1
        self.deck_index = 5

        for u in range(num_vertices):
            for v in range(num_vertices):
                for c in range(num_colors):
                    if (u, v, c) in edges:
                        self.graph[u][v][c] = edges[(u, v, c)]
                    else:
                        self.status[u][v][c] = -1

        

    def claim_route(self, route, player):
        """
        claim a route using train cards on a player's hand 
        route: must be valid and available
        cards: will be modified after train cards are used
        """
        u,v,c = route
        assert u < v
        self.status[u][v][c] = 1
        count = self.graph[u][v][c]
        player.trains -= count

        if c == 0:
            c = np.argmax(player.cards[1:]) 
        if c == 9:
            player.cards[0] -= 1
            count -= 1
            c = np.argmax(player.cards[1:])

        if player.cards[c] >= count:
            player.cards[c] -= count
        else:
            player.cards[0] -= count - player.cards[c]
            player.cards[c] = 0

    def take_card(self, c, player):
        """
        take 1 public card of color c and replace it with the top card in the deck
        """
        assert sum(self.public_cards) == 5
        player.cards[c] += 1
        self.public_cards[c] -= 1
        self.public_cards[self.deck_cards[self.deck_index]] += 1
        self.deck_index += 1


    def get_graph(self):
        return self.graph

    def get_status(self):
        return self.status


