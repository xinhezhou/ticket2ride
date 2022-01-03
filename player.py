import numpy as np
class Player:
    def __init__(self):
        self.cards = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    def choose_route(self, graph, availability, public_cards):
        pass

    def choose_card(self, graph, availability, public_cards):
        available_colors = []
        for i in range(9):
            if public_cards[i] > 0:
                available_colors.append(i)
        return np.random.choice(available_colors)