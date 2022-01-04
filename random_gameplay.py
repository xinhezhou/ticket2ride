from game import Game
from random_player import Player
from game_utils import get_possible_routes, get_route_score, check_path, compute_availability_matrix

import matplotlib.pyplot as plt

num_vertices = 6
num_colors = 10
edges = {
    (0, 2, 1): 1,
    (0, 3, 8): 3,
    (0, 4, 7): 2,
    (1, 3, 2): 2,
    (1, 4, 1): 3,
    (1, 4, 6): 3,
    (2, 4, 2): 2,
    (3, 4, 3): 2,
    (3, 5, 9): 3,
}
num_trains = 10
iterations = 1000




def play_game(iterations):
    scores = []
    rounds = []
    path_complete = []

    for _ in range(iterations):
        game = Game(num_vertices, num_colors, edges)
        player = Player(num_trains)
        graph = game.get_graph()
        num_rounds = 0
        while player.trains > 2:

            c = player.choose_card(graph, game.status, game.public_cards)
            game.take_card(c, player)
            if c != 0:
                c = player.choose_card(graph, game.status, game.public_cards)
                game.take_card(c, player)

            route  = player.choose_route(graph, compute_availability_matrix(graph, game.status, player.cards))
            if route is not None:
                game.claim_route(route, player)
            num_rounds += 1

        rounds.append(num_rounds)
        score = 0
        for i in range(game.v):
            for j in range(game.v):
                for k in range(game.c):
                    if game.status[i][j][k] == 1:
                        score += get_route_score(game.graph, (i, j, k))
        complete = check_path(game.status, 4, 5)
        if complete:
            score += 5
            path_complete.append(1)
        else:
            path_complete.append(0)
        scores.append(score)
    return scores, rounds, path_complete


scores, rounds, path_complete  = play_game(iterations)
print(scores)
print(rounds)
print(path_complete)


fig, ax = plt.subplots(3)
ax[0].hist(scores, density=False, bins=10)
ax[0].title.set_text("scores")
ax[1].hist(rounds, density=False, bins=10)
ax[1].title.set_text("rounds")
ax[2].hist(path_complete, density=False, bins=2)
ax[2].title.set_text("path_complete")
fig.tight_layout()
plt.savefig("diagrams/random_single_player.pdf")
# plt.show()




