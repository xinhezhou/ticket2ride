import sys
sys.path.append("../../../")
import numpy as np
from game import Game
from nonRLplayers.random_player import RandomPlayer
from utils.game_utils import check_win, compute_availability_matrix, get_available_routes

    

def simulate_rollout(game, players, action):
    new_game = game.duplicate()
    new_players = [player.duplicate() for player in players]
    player = new_players[0]
    if action ==  0:
        new_game.draw_cards(player)
    else:
        new_game.claim_route(action, player)
    winner = check_win(game, players)
    count = 0
    while winner == -1:
        count += 1
        player = players[0]
        availability = compute_availability_matrix(game.graph, game.status, player)
        if len(get_available_routes(availability)) == 0 or player.draw_or_claim(game, players) == 0:
            cards = game.draw_cards(player)
        else:
            route = player.choose_route(game, players) 
            game.claim_route(route, player)

        players = players[::-1]
        winner = check_win(game, players)
        if count > 100:
            print(winner, player.cards)
    return winner

num_vertices = 7
num_route_colors = 7
num_card_colors = 7
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
deck_cards = [1,2,3,4,5,6,0] * 10  + [0] * 2 # train cards in the deck


game = Game(num_vertices, num_route_colors, edges, deck_cards)
player_a = RandomPlayer(num_card_colors, [(1, 3), (1, 6)], 10, 1)
player_b = RandomPlayer(num_card_colors, [(0, 6), (3, 6)], 10, 2)

winner = simulate_rollout(game, [player_a, player_b], 0)
# print(winner)