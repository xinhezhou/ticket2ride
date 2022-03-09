
import json 
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
from bot_gameplay import play_game



def compute_route_freq(records, k):
    route_freq = {}
    win_freq = {}
    for key in records:
        record = records[key]
        actions = record["actions"]
        winner = tuple(record["winner"])
        if winner in win_freq:
            win_freq[winner] += 1
        else:
            win_freq[winner] = 1
        # print(actions[0])
        routes = []
        for i in range(0, len(actions), 2):
            action = actions[i]
            if len(action) == 3:
                routes.append(tuple(action[:2]))
        routes.sort()
        if tuple(routes) in route_freq:
            route_freq[tuple(routes)] += 1
        else:
            route_freq[tuple(routes)] = 1

    for key in route_freq:
        if route_freq[key] > k:
            print(key, route_freq[key])
    print(sorted(route_freq.values()))
    print(win_freq)
    return win_freq, route_freq

def plot_winners(initialize_game, round=1000):
    winners, records = play_game(1000, initialize_game)
    print(winners)
    with open("cnn_random_lookahead.json", "w") as outfile:
        json.dump(records, outfile)
    plt.hist(winners, density=False, bins=3,)
    plt.show()

f = open("cnn_two_shuffled.json")
records = json.load(f)
win_freqs, route_freqs = compute_route_freq(records, 1)

