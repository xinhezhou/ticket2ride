
import json 
import sys
import matplotlib.pyplot as plt



def compute_route_freq(records, k, win_file=None):
    route_freq = {}
    win_freq = {}
    for key in records:
        record = records[key]
        actions = record["actions"]
        winner = str(record["winner"])
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
    if win_file:
        plt.clf()
        # plt.hist(win_freq)
        keys = sorted(win_freq.keys())
        values = [win_freq[key] for key in keys]
        plt.bar(keys, values, width=.5, color='b')
        plt.savefig(win_file)
    return win_freq, route_freq


filenames = [
    ["../rl/with_prior_es.pth.tar", "with_prior_es_random.json"],
    ["../rl/without_prior_es.pth.tar", "with_prior_es_random.json"],
]
for filename, win_file in filenames:
    print(filename)
    f = open(filename)
    records = json.load(f)
    win_freqs, route_freqs = compute_route_freq(records, 30, win_file)


