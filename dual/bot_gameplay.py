import sys
sys.path.append("..")
from utils.game_utils import check_win, compute_availability_matrix, get_available_routes



def play_game(iterations, initialize_game):
    """
    Simulate gameplay and record number of rounds and trains used each time
    """
    winners = []
    records = {}

    for i in range(iterations):
        record = {}
        player_index = 0
        game, players = initialize_game()
        record["deck"] = game.cards
        record["destinations_a"] = players[0].destination_cards
        record["destinations_b"] = players[1].destination_cards
        actions = []
        winner = -1
        while winner == -1:
            player = players[player_index]
            availability = compute_availability_matrix(game.graph, game.status, player)
            if len(get_available_routes(availability)) == 0 or player.draw_or_claim(game) == 0:
                cards = game.draw_cards(player)
                actions.append(cards)
            else:
                route = player.choose_route(game) 
                game.claim_route(route, player)
                actions.append(route)

            player_index = (player_index + 1) % 2
            winner = check_win(game, players)

        record["actions"] = actions
        record["winner"] = winner
        winners.append(winner)
        # if winner == 0:
        #     print(players[0].routes)
        #     print(players[1].routes)
        records[i] = record



    
    return winners, records
    
    
# # print(winners, trains_a, trains_b)
# fig, ax = plt.subplots(4)
# ax[0].hist(trains_a, density=False, bins=8)
# ax[0].title.set_text("A trains left")
# ax[1].hist(trains_b, density=False, bins=8)
# ax[1].title.set_text("B trains left")
# ax[2].hist(rounds, density=False, bins=10)
# ax[2].title.set_text("number of rounds")
# ax[3].hist(winners, density=False, bins=3,)
# ax[3].title.set_text("winner")
# fig.tight_layout()
# plt.show()





