import json
import sys
sys.path.append("../")
from cnn_utils import generate_state_matrix
sys.path.append("../")
from game import Game
from nonRLplayers.random_player import RandomPlayer

def generate_gameplay(records):
    """
    Generate gameplays from records and push them into memory
    for CNN training
    """
    num_vertices = 7
    num_route_colors = 7
    num_card_colors = 7
    edges = {
        (0, 1, 2): 1,
        (0, 3, 2): 3,
        (0, 3, 5): 3,
        (0, 4, 6): 2,
        (1, 2, 6): 2,
        (1, 4, 1): 2,
        (1, 4, 2): 2,
        (2, 4, 1): 1,
        (2, 4, 3): 1,
        (2, 6, 6): 2,
        (3, 4, 4): 2,
        (3, 5, 4): 1,
        (3, 5, 6): 1,
        (4, 5, 3): 2,
        (4, 5, 6): 2,
        (4, 6, 6): 3,
        (5, 6, 2): 3,
        (5, 6, 3): 3,
    }
    memory = {}
    for key in records:
        record = records[key]
        deck_cards = record["deck"]
        game = Game(num_vertices, num_route_colors, edges, deck_cards)
        player_a = RandomPlayer(num_card_colors, record["destinations_a"], 10, 1)
        player_b = RandomPlayer(num_card_colors, record["destinations_b"], 10, 2)
        players = [player_a, player_b]
        for action in record["actions"]:
            reward = 0
            state = generate_state_matrix(game, players).unsqueeze(0)
            player = players[0]
            if len(action) == 2:
                choice = 0
                game.draw_cards(player)
            else:
                u, v, c = action
                choice = 1 + u*49 + v*7 + c
                game.claim_route(action, player)
                if action == record["actions"][-1]:
                    reward = 1
            next_state = generate_state_matrix(game, players).unsqueeze(0)
            if player.id == 1:
                entry = {}
                entry["state"]  = state.tolist()
                entry["choice"] = [[choice]]
                entry["next_state"] = next_state.tolist()
                entry["reward"] = [reward]
                memory[key] = entry
            players = players[::-1]
    return memory

def convert_records(json_file, memory_file):
    f = open(json_file)
    records = json.load(f)
    records = generate_gameplay(records)
    with open(memory_file, "w") as outfile:
        json.dump(records, outfile)

# filenames = [
#     ["small.json", "small_memory.json"],
#     ["medium.json", "medium_memory.json"],
#     ["large.json", "large_memory.json"]
# ]
# for record_file, memory_file in filenames:
#     convert_records(record_file, memory_file)

