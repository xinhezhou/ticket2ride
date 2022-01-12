from game import Game
from players.random_player import Player
from game_utils import get_possible_routes
from translate_utils import translate_cards, translate_route

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
player = Player(10)

def test_draw_cards(game, player, count):
    graph = game.get_graph()
    for i in range(count):
        print(translate_cards(player.cards))
        availability = game.compute_route_availability(player)
        print(get_possible_routes(num_vertices, num_colors, availability))

        c = player.choose_card(graph, availability, game.public_cards)
        game.take_card(c, player)
        print("\n")


def test_claim_routes(game, player, count):
    graph = game.get_graph()
    for i in range(count):
        # print(translate_cards(player.cards))

        c = player.choose_card(graph, game.compute_route_availability(player), game.public_cards)
        game.take_card(c, player)

        route  = player.choose_route(graph, game.compute_route_availability(player))
        if route is not None:
            print(translate_cards(player.cards))
            print(get_possible_routes(num_vertices, num_colors, game.compute_route_availability(player)))
            print("taken: ", translate_route(route))
            game.claim_route(route, player)
            print(translate_cards(player.cards))
            print("\n")


test_claim_routes(game, player, 1)
# player.cards = [1, 1, 0, 0, 0, 0, 1, 1, 1]
# availability = game.compute_route_availability(player)
# print(get_available_routes(num_vertices, num_colors, availability))
