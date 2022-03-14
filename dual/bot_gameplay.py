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
        game, players = initialize_game()
        record["deck"] = game.cards
        record["destinations_a"] = players[0].destination_cards
        record["destinations_b"] = players[1].destination_cards
        actions = []
        winner = -1, -1
        count = 0
        while winner[0] == -1:
            count += 1
            player = players[0]
            availability = compute_availability_matrix(game.graph, game.status, player)
            if len(get_available_routes(availability)) == 0 or player.draw_or_claim(game, players) == 0:
                cards = game.draw_cards(player)
                actions.append(cards)
            else:
                route = player.choose_route(game, players) 
                game.claim_route(route, player)
                actions.append(route)
                # if player.id == 1:
                #     print(route)
            winner = check_win(game, players)
            if count > 100:    
                print(winner)
            players = players[::-1]

        record["actions"] = actions
        record["winner"] = winner
        winners.append(winner[0])
        # if winner == 0:
        # print(players[0].routes, players[1].routes, winner)
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





