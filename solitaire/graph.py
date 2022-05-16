from select import POLLOUT
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



# LOCATION_B = {
#     0: (10, 2),
#     1: (11, 3.73),
#     2: (10, 5.46),
#     3: (8, 2),
#     4: (9, 3.73),
#     5: (7, 3.73),
#     6: (8,5.46),
# }



def compute_route_freq(records, k, win_file=None):
    route_freq_a = {}
    counter = 0
    total = 0
    for key in records:
        record = records[key]
        actions = record["actions"]
        # print(actions)
        routes_a = []
        for i in range(0, len(actions)):
            total += 1
            action = actions[i]
            if len(action) == 3:
                counter += 1
                routes_a.append(tuple(action))
        routes_a.sort()
        if tuple(routes_a) in route_freq_a:
            route_freq_a[tuple(routes_a)] += 1
        else:
            route_freq_a[tuple(routes_a)] = 1

    # print("attacker")
    routes_a = {}
    for key in route_freq_a:
        if route_freq_a[key] > k:
            # print(key, route_freq_a[key])
            for route in key:
                if route in routes_a:
                    routes_a[route] += route_freq_a[key]
                else:
                    routes_a[route] = route_freq_a[key]
    print(routes_a)
    print(counter/total)

    # for route in routes_a:
    #     p1 = LOCATION_A[route[0]]
    #     p2 = LOCATION_A[route[1]]
    #     shade_line(p1, p2, routes_a[route], u'#1f77b4' )

    # print("defender")
    # routes_b = {}
    # for key in route_freq_b:
    #     if route_freq_b[key] > k:
    #         print(key, route_freq_b[key])
    #         for route in key:
    #             if route in routes_b:
    #                 routes_b[route] += route_freq_b[key]
    #             else:
    #                 routes_b[route] = route_freq_b[key]
    # for route in routes_b:
    #     p1 = LOCATION_B[route[0]]
    #     p2 = LOCATION_B[route[1]]
    #     shade_line(p1, p2, routes_b[route], u'#ff7f0e')

    # print(sorted(route_freq_b.values()))


    return routes_a


def shade_line(p1, p2, freq, color):
    x_1, x_2 = p1
    y_1, y_2 = p2
    for j in range(freq+1):
        plt.plot(x_1 + (x_2-x_1)*j/(freq+1), y_1 + (y_2 - y_1)*j/(freq),
                 alpha=0.2,
                 markersize = 8,
                 marker = "o", 
                 color= color)


# players = ["SupervisedPG", "SelfplayPG", "SupervisedES", "SelfplayES", "Random", "Greedy"]
plt.clf()
# plt.xlim(1, 7)
# plt.ylim(1, 7)
plt.axis("off")
EDGE_MAP = {
  (0, 1, 2): 2,
  (0, 3, 2): 0,
  (0, 3, 5): 1,
  (0, 4, 6): 3,
  (1, 2, 6): 7,
  (1, 4, 1): 4,
  (1, 4, 2): 5,
  (2, 4, 1): 13,
  (2, 4, 3): 12,
  (2, 6, 6): 8,
  (3, 4, 4): 6,
  (3, 5, 4): 15,
  (3, 5, 6): 14,
  (4, 5, 3): 11,
  (4, 5, 6): 10,
  (4, 6, 6): 9,
  (5, 6, 2): 16,
  (5, 6, 3): 17,
}

edges = [
([3, 5], [2.1, 2.1,], 'b'),
([3, 5], [1.9, 1.9], 'm'),
([5, 6], [2, 3.73], 'b'),
([5, 4], [2, 3.73], 'y'),
([4, 6], [3.83, 3.83], "k"),
([4, 6], [3.63, 3.63], "b"),
([4, 3], [3.73, 2], 'r' ),
([6, 5], [3.73, 5.46], 'y'),
([3, 5], [5.46, 5.46], 'y'),
([3, 4], [5.46, 3.73], 'y'),
([2, 4], [3.83, 3.83], "y"),
([2, 4], [3.63, 3.63], "g"),
([4.05, 5.05], [3.63, 5.46], "g"),
([3.9, 4.87], [3.72, 5.46], "k"),
([3.1, 2.1], [2, 3.73], "y"),
([2.9, 1.9], [2, 3.73], "r"),
([3.1, 2.1], [5.46, 3.73], "b"),
([2.9, 1.9], [5.46, 3.73], "g"),
]

vertices = [
    (3, 2, "3"),
    (5, 2, "0"),
    (6, 3.73, "1"),
    (5, 5.46, "2"),
    (3, 5.46, "6"),
    (2, 3.73, "5"),
    (4, 3.73, "4")
]

special_marks = [
    (6, 3.73, "b"),
    (2, 3.73, "b"),
    (3, 2, "r"),
    (5,5.46, "r"),
    # (5, 2, "g"),
    # (3, 5.46, "g"),
]
for key in EDGE_MAP:
    x, y, c = edges[EDGE_MAP[key]]
    # print(x, y, c)
    plt.plot(x, y, color=c)

for x, y, s in vertices:
    plt.plot(x, y, marker = "o", markersize = 20, color = 'white', markeredgecolor = 'k')
    plt.text(x-0.05, y-0.09, s)

for x, y, c in special_marks:
    plt.plot(x, y, marker = "o", markersize = 20, color = c, alpha=0.3)


# corners_x = [3, 5, 6, 5, 3, 2, 3]
# corners_y = [2, 2, 3.73, 5.46, 5.46, 3.73, 2]
# cities = ["3", "0", "1", "2", "6", "5"]
# center = (4, 3.73)
# # draw hexagon 1
# plt.plot(corners_x, corners_y, color='k')
# # draw routes
# for i in range(6):
#     x, y = corners_x[i], corners_y[i]
#     plt.plot([center[0], x], [center[1], y], color='k')
#     plt.plot([center[0]-0.05, x-0.05], [center[1]+0.05, y+0.05], color='k')
#     plt.plot(x, y, marker = "o", markersize = 20, color = 'white', markeredgecolor = 'k')
#     plt.text(x-0.05, y-0.09, cities[i])
# plt.plot(center[0], center[1], marker = "o", markersize = 20, color = 'white', markeredgecolor = 'k')
# plt.text(center[0]-0.05, center[1]-0.09, "4")

f = open("dqn_eval_record_3.json")
records = json.load(f)
routes = compute_route_freq(records, 5, win_file=None)
print(routes)
for route in routes:
    freq = routes[route]
    p1, p2, color = edges[EDGE_MAP[route]]
    shade_line(p1, p2, freq, color)

plt.show()
