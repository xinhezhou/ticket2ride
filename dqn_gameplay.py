from game import Game
from dqn_player import Player
from game_utils import get_possible_routes, get_route_score, check_path, compute_availability_matrix
from dqn_network import Network
from dqn_utils import compute_card_agent_input
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

from translate_utils import translate_route

num_vertices = 6
num_route_colors = 10
num_card_colors = 9
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

num_inputs = (num_vertices * num_vertices * num_route_colors + num_card_colors) * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
card_agent = Network(num_inputs, num_card_colors, 100, device)

gamma = 0.9
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(card_agent.parameters())
colors = [0,0,0,0,0,0,0,0,0]



def play_game(iterations):
    scores = []
    path_complete = []

    for _ in range(iterations):
        game = Game(num_vertices, num_route_colors, edges)
        player = Player(num_trains, card_agent)
        graph = game.get_graph()
        while player.trains > 2:
            prev_cards = player.cards[:]
            state_action_values, c1 = player.choose_card(graph, game.status, game.public_cards)
            
            game.take_card(c1, player)
            if c1 != 0:
                __, c2 = player.choose_card(graph, game.status, game.public_cards)
                if c2 != 0:
                    game.take_card(c2, player)
                    colors[c2] += 1
            colors[c1] += 1

            reward = 0
            prev_path_complete = check_path(game.status,4,5)
            route  = player.choose_route(graph, compute_availability_matrix(graph, game.status, player.cards))
            if route is not None:
                game.claim_route(route, player)
                reward += get_route_score(graph, route)
                if check_path(game.status,4,5) is not prev_path_complete:
                    reward += 5

            reward = reward / 16
            if state_action_values is not None:
                next_input = compute_card_agent_input(graph, game.status, game.public_cards, player.cards)
                next_best_q = max(card_agent(next_input))
                expected_state_action_avlue = reward + gamma * next_best_q

                loss = criterion(state_action_values[c1], expected_state_action_avlue)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(player.cards, state_action_values)
            


        score = 0
        routes = []
        for i in range(game.v):
            for j in range(game.v):
                for k in range(game.c):
                    if game.status[i][j][k] == 1:
                        score += get_route_score(game.graph, (i, j, k))
                        routes.append(translate_route((i,j,k)))
        # print(routes)
        complete = check_path(game.status, 4, 5)
        if complete:
            score += 5
            path_complete.append(1)
        else:
            path_complete.append(0)
        scores.append(score)
    return scores, path_complete


scores, path_complete  = play_game(iterations)
print(path_complete)



fig, ax = plt.subplots(2)
ax[0].hist(scores, density=False, bins=10)
ax[0].title.set_text("scores")
ax[1].hist(path_complete, density=False, bins=2)
ax[1].title.set_text("path_complete")
fig.tight_layout()
# plt.savefig("diagrams/random_single_player.pdf")
plt.show()




