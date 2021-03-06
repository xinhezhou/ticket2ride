from matplotlib.pyplot import jet
import numpy as np

scoring_table = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
    4: 7,
}


def compute_availability_matrix(graph, status, player):
    """
    compute an availbility binary matrix based on the status of the routes 
    and the cards owned by the player
    """
    availability = np.zeros(status.shape)
    for u in range(len(availability)):
        for v in range(len(availability)):
            for c in range(len(availability[0][0])):
                if (u, v) not in player.routes and graph[u][v][c] > 0 and  status[u][v][c] == 0 and player.cards[0] + player.cards[c] >= graph[u][v][c] and player.trains >= graph[u][v][c]:
                    availability[u][v][c] = 1
    return availability

def get_available_routes(availability):
    """
    return all available routes from an availability matrix
    """
    routes = []
    v = len(availability)
    c = len(availability[0][0])
    for i in range(v):
        for j in range(v):
            for k in range(c):
                if availability[i][j][k] == 1:
                    routes.append(((i,j,k)))
    return routes
    


def compute_progress (graph, status, route, destination_cards, id):
    """
    Use Dijkstra to compute the progress made by taking the route
    progress is measured by the change in distance between
    the origin and the destination, where claimed routes now have
    0 distance
    """
    # compute graph weights based on train placement
    n = len(graph)
    g = -1 * np.ones((n, n))
    for i in range(len(graph)):
        for j in range(len(graph)):
            if max(graph[i][j]) != 0:
                if id in status[i][j]:
                    g[i][j] = 0
                else:
                    # print(g)
                    g[i][j] = max(graph[i][j])
            
    d1 = 0
    for s, t in destination_cards:
        d1 += run_Dijkstra(g, s, t, n)
    u,v,c = route
    g[(u, v)] = 0 # the claimed route now has 0 weight
    d2 = 0
    for s, t in destination_cards:
        d2 += run_Dijkstra(g, s, t, n)
    return d1 - d2


def run_Dijkstra(graph, s, t, n):
    """
    Run Dijkstra's algorithm on a graph with nonnegative weights
    to find the distance of the shortest path between 
    """
    queue = {}
    for i in range(n):
        queue[i] = 100 # arbitrary big number
    distances = [100] * n
    queue[s] = 0
    distances[s] = 0
    while len(queue) > 0:
        # print(queue)
        min_vertex = list(queue.keys())[0]
        min_d = queue[min_vertex]
        # find closest vertex
        for v in queue:
            if queue[v] < min_d:
                min_d = queue[v]
                min_vertex = v
        # relax outgoing edges
        for i in range(n):
            if graph[i][min_vertex] != -1 or graph[min_vertex][i] != -1:
                # print(graph)
                weight = max(graph[i][min_vertex], graph[min_vertex][i])
                distances[i] = min(distances[i], distances[min_vertex] + weight)
                if i in queue:
                    queue[i] = distances[i]
        queue.pop(min_vertex)

    # print(graph,distances)
    return distances[t]


def check_path(status, a, b, id=1):
    """
    use BFS to check if there a path between a and b
    """
    visited = {a}
    current = {a}
    while len(current) != 0:
        next_level = set()
        for u in current:
            for v in range(len(status)):
                for c in range(len(status[0][0])):
                    if v not in visited and (status[u][v][c] == id or status[v][u][c] == id):
                        next_level.add(v)
                        visited.add(v)
        current = next_level
    return b in visited


def check_path(status, a, b, id=1):
    """
    use BFS to check if there a path between a and b
    """
    visited = {a}
    current = {a}
    while len(current) != 0:
        next_level = set()
        for u in current:
            for v in range(len(status)):
                for c in range(len(status[0][0])):
                    if v not in visited and (status[u][v][c] == id or status[v][u][c] == id):
                        next_level.add(v)
                        visited.add(v)
        current = next_level
    return b in visited

def check_win(game, players):
    """
    -1: ongoing
    0: tie
    > 0: id of winner
    """
    # print(status)
    # print(status)
    for player in players:
        if len(player.destination_cards) == 0:
            return player.id, 1
        if game.card_index >= len(game.cards) or player.trains == 0:
            if len(players[0].destination_cards) == len(players[1].destination_cards):
                # print(player.routes, player.destination_cards)
                return 0, 0
            elif len(players[0].destination_cards) > len(players[1].destination_cards):
                return players[1].id, 0
            else:
                return players[0].id, 0
    
    for u in range(len(game.status)):
        for v in range(len(game.status)):
            if 0 in game.status[u][v]:
                return -1, -1

    # print(status)
    return 0
    


