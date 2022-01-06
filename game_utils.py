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
                if (u, v) not in player.routes and graph[u][v][c] > 0 and  status[u][v][c] == 0 and player.cards[0] + player.cards[c] >= graph[u][v][c]:
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
    

def check_path(status, a, b):
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
                    if v not in visited and (status[u][v][c] == 1 or status[v][u][c] == 1):
                        next_level.add(v)
                        visited.add(v)
        current = next_level
    return b in visited


