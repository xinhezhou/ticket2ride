cities = {
    0: "Amsterdam",
    1: "Berlin",
    2: "Bruxelles",
    3: "Essen",
    4: "Frankfurt",
    5: "Kobenhavn",
}


route_colors = {
    0: "grey",
    1: "black",
    2: "blue",
    3: "green",
    4: "orange",
    5: "purple",
    6: "red",
    7: "white",
    8: "yellow",
    9: "1 rainbow",
}

card_colors = {
    0: "rainbow",
    1: "black",
    2: "blue",
    3: "green",
    4: "orange",
    5: "purple",
    6: "red",
    7: "white",
    8: "yellow",
}


def get_available_routes(num_vertices, num_colors, availability):
    routes = []
    for u in range(num_vertices):
        for v in range(num_vertices):
            for c in range(num_colors):
                if availability[u][v][c]:
                    routes.append(translate_route(u,v,c))
    return routes


def translate_route(u, v, c):
    return (cities[u], cities[v], route_colors[c])

def translate_cards(cards):
    cards_by_color = []
    for i in range(9):
        cards_by_color.append((card_colors[i], cards[i]))
    return cards_by_color

