from game import Game
from player import Player
from utils import get_available_routes, translate_cards, translate_route

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

game = Game(num_vertices, num_colors, edges)
player = Player()

def test_draw_cards(game, player, count):
    graph = game.get_graph()
    for i in range(count):
        print(translate_cards(player.cards))
        availability = game.compute_route_availability(player)
        print(get_available_routes(num_vertices, num_colors, availability))

        c = player.choose_card(graph, availability, game.public_cards)
        game.take_card(c, player)
        print("\n")


test_draw_cards(game, player,10)
