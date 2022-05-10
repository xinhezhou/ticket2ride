import matplotlib.pyplot as plt
import numpy as np
import json

LOCATION_A = {
    0: (5, 2),
    1: (6, 3.73),
    2: (5, 5.46),
    3: (3, 2),
    4: (4, 3.73),
    5: (2, 3.73),
    6: (3,5.46),
}



LOCATION_B = {
    0: (10, 2),
    1: (11, 3.73),
    2: (10, 5.46),
    3: (8, 2),
    4: (9, 3.73),
    5: (7, 3.73),
    6: (8,5.46),
}



def compute_route_freq(records, k, win_file=None):
    route_freq_a = {}
    route_freq_b = {}
    for key in records:
        record = records[key]
        actions = record["actions"]
        routes_a = []
        routes_b = []
        for i in range(0, len(actions)):
            action = actions[i]
            if len(action) == 3:
                if i % 2 == 0:
                    routes_a.append(tuple(action[:2]))
                else:
                    routes_b.append(tuple(action[:2]))
        routes_a.sort()
        routes_b.sort()
        if tuple(routes_a) in route_freq_a:
            route_freq_a[tuple(routes_a)] += 1
        else:
            route_freq_a[tuple(routes_a)] = 1
        
        if tuple(routes_b) in route_freq_b:
            route_freq_b[tuple(routes_b)] += 1
        else:
            route_freq_b[tuple(routes_b)] = 1

    print("attacker")
    routes_a = {}
    for key in route_freq_a:
        if route_freq_a[key] > k:
            print(key, route_freq_a[key])
            for route in key:
                if route in routes_a:
                    routes_a[route] += route_freq_a[key]
                else:
                    routes_a[route] = route_freq_a[key]
    print(sorted(route_freq_a.values()))

    for route in routes_a:
        p1 = LOCATION_A[route[0]]
        p2 = LOCATION_A[route[1]]
        shade_line(p1, p2, routes_a[route], u'#1f77b4' )

    print("defender")
    routes_b = {}
    for key in route_freq_b:
        if route_freq_b[key] > k:
            print(key, route_freq_b[key])
            for route in key:
                if route in routes_b:
                    routes_b[route] += route_freq_b[key]
                else:
                    routes_b[route] = route_freq_b[key]
    for route in routes_b:
        p1 = LOCATION_B[route[0]]
        p2 = LOCATION_B[route[1]]
        shade_line(p1, p2, routes_b[route], u'#ff7f0e')

    print(sorted(route_freq_b.values()))


    return route_freq_a, route_freq_b


def shade_line(p1, p2, freq, color):
    x_1, y_1 = p1
    x_2, y_2 = p2
    for j in range(freq+1):
        plt.plot(x_1 + (x_2-x_1)*j/(freq+1), y_1 + (y_2 - y_1)*j/(freq),
                 alpha=0.2,
                 markersize = 8,
                 marker = "o", 
                 color= color)


# players = ["SupervisedPG", "SelfplayPG", "SupervisedES", "SelfplayES", "Random", "Greedy"]
plt.figure(figsize = (12, 8))
title =  "PGSupervisedHideTraining"
f = open(title + ".json")
records = json.load(f)
plt.clf()
compute_route_freq(records, 20, win_file=None)

corners_x = [3, 5, 6, 5, 3, 2, 3]
corners_y = [2, 2, 3.73, 5.46, 5.46, 3.73, 2]
center = (4, 3.73)
# draw hexagon 1
plt.plot(corners_x, corners_y, color='k')
# draw routes 
for i in range(6):
    x, y = corners_x[i], corners_y[i]
    plt.plot([center[0], x], [center[1], y], color='k')

corners_x = [8, 10, 11, 10, 8, 7, 8]
corners_y = [2, 2, 3.73, 5.46, 5.46, 3.73, 2]
center = (9, 3.73)
# draw hexagon 2
plt.plot(corners_x, corners_y, color='k')
# draw routes 
for i in range(6):
    x, y = corners_x[i], corners_y[i]
    plt.plot([center[0], x], [center[1], y], color='k')

plt.savefig(title + '_route.pdf') 

plt.xlim(1, 12)
plt.ylim(1, 8)
# plt.show()