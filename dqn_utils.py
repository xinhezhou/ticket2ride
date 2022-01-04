from game_utils import get_possible_routes, get_route_score, compute_availability_matrix
import numpy as np
import torch


def compute_card_agent_input(graph, status, public_cards, player_cards):
    availability = compute_availability_matrix(graph, status, player_cards)
    array = np.concatenate([graph.flatten(), availability.flatten(), public_cards, player_cards])
    return torch.from_numpy(array).float()
