cities = {
    0: "Albuquerque",
    1: "Denver",
    2: "Helena",
    3: "Los Angeles",
    4: "Salt Lake City",
    5: "San Francisco",
    6: "Seattle",
}


route_colors = {
    1: "black",
    2: "blue",
    3: "green",
    4: "red",
    5: "white",
    6: "yellow",
}

card_colors = {
    0: "rainbow",
    1: "black",
    2: "blue",
    3: "green",
    4: "red",
    5: "white",
    6: "yellow"
}

def translate_route(route):
    u,v,c = route
    return (cities[u], cities[v], route_colors[c])

def translate_cards(cards):
    cards_by_color = []
    for i in range(9):
        cards_by_color.append((card_colors[i], cards[i]))
    return cards_by_color

