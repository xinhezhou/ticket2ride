import numpy as np

class Player:
    def __init__(self, num_trains, model=None):
        self.cards = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.trains = num_trains

    def choose_route(self, graph, availability):
        if np.random.random_sample() < 0.2:
            return None

        available_routes = []
        v = len(availability)
        c = len(availability[0][0])
        for i in range(v):
            for j in range(v):
                for k in range(c):
                    if availability[i][j][k] and self.trains > graph[i][j][k]:
                        available_routes.append((i,j,k))

        if len(available_routes) == 0:
            return None
        else:
            return available_routes[np.random.randint(len(available_routes))]

    def choose_card(self, graph, status, public_cards):
        available_colors = []
        for i in range(9):
            if public_cards[i] > 0:
                available_colors.append(i)
        return np.random.choice(available_colors)