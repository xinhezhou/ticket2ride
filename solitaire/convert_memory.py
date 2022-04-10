import json
import sys
sys.path.append("../")
from RLplayers.rl_utils import generate_state_matrix
sys.path.append("../")
from game import Game
from nonRLplayers.random_player import RandomPlayer
from utils.game_utils import compute_progress

def generate_gameplay(records, input_f=generate_state_matrix):
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
    counter = 0
    for key in records:
        record = records[key]
        deck_cards = record["deck"]
        game = Game(num_vertices, num_route_colors, edges, deck_cards)
        player = RandomPlayer(num_card_colors, record["destinations"], 10, 1)
        players = [player]
        for action in record["actions"]:
            counter += 1
            state = input_f(game, players).unsqueeze(0)
            player = players[0]
            if len(action) == 2:
                choice = 0
                game.draw_cards(player)
                reward = 0
            else:
                u, v, c = action
                choice = 1 + u*49 + v*7 + c
                reward = compute_progress(game.graph, game.status, action, player.destination_cards, player.id)
                game.claim_route(action, player)
            next_state = input_f(game, players).unsqueeze(0)
            # print(next_state.shape)
            if player.id == 1:
                entry = {}
                entry["state"]  = state.tolist()
                entry["choice"] = [[choice]]
                entry["next_state"] = next_state.tolist()
                entry["reward"] = [reward]
                memory[counter] = entry
            players = players[::-1]
    return memory

def convert_records(json_file, memory_file, input_f=generate_state_matrix):
    f = open(json_file)
    records = json.load(f)
    records = generate_gameplay(records,input_f)
    with open(memory_file, "w") as outfile:
        json.dump(records, outfile)
        
if __name__ == '__main__':
    record_file = "dqn_record.json"
    memory_file = "dqn_memory.json"
    convert_records(record_file, memory_file)

