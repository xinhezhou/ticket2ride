from game import Game
from random_player import RandomPlayer
from game_utils import check_path, compute_availability_matrix
from translate_utils import translate_route
import numpy as np
import matplotlib.pyplot as plt

num_vertices = 7
num_route_colors = 6
num_card_colors = 7
deck_cards = [0] * 12  + [1,2,3,4,5,6] * 10 # train cards in the deck
edges = {
  (0, 1, 2): 1,
  (0, 3, 2): 3,
  (0, 3, 5): 3,
  (0, 4, 6): 2,
  (1, 2, 6): 2,
  (1, 4, 1): 2,
  (1, 4, 2): 2,
  (2, 4, 1): 1,
  (2, 4, 3): 1,
  (2, 6, 6): 2,
  (3, 4, 4): 2,
  (3, 5, 4): 1,
  (3, 5, 6): 1,
  (4, 5, 3): 2,
  (4, 5, 6): 2,
  (4, 6, 6): 3,
  (5, 6, 2): 3,
  (5, 6, 3): 3,
}
start = 0
end = 6
iterations = 1000




def play_game(iterations):
    trains = []
    rounds = []

    for _ in range(iterations):
        np.random.shuffle(deck_cards) 
        game = Game(num_vertices, num_route_colors, edges, deck_cards)
        player = RandomPlayer(num_card_colors)
        game.draw_cards(player)
        game.draw_cards(player)
        num_rounds = 0
        while not check_path(game.status, start, end):
            num_rounds += 1
            availability = compute_availability_matrix(game.graph, game.status, player)
            if game.card_index < len(game.cards) and player.draw_or_claim(game.graph, availability) == 0:
                game.draw_cards(player)
            else:
                availability = compute_availability_matrix(game.graph, game.status, player)
                route = player.choose_route(game.graph, availability)
                if route is not None:
                    game.claim_route(route, player)

        rounds.append(num_rounds)
        trains.append(player.trains_used)
        routes = []
        for u, v in player.routes:
            routes.append(translate_route((u,v,player.routes[(u,v)])))
        # print(routes)

    return trains, rounds


trains, rounds = play_game(iterations)
print(trains)
print(rounds)


fig, ax = plt.subplots(2)
ax[0].hist(trains, density=False, bins=8)
ax[0].title.set_text("trains used")
ax[1].hist(rounds, density=False, bins=10)
ax[1].title.set_text("number of rounds")
fig.tight_layout()
plt.savefig("diagrams/solitaire_random.pdf")
# plt.show()



