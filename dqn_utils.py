from game_utils import get_possible_routes, get_route_score, compute_availability_matrix
import numpy as np
import torch


def compute_reward(graph, status, prev_cards, next_cards):
    prev_availability = compute_availability_matrix(graph, status, prev_cards)
    prev_routes = get_possible_routes(prev_availability)
    prev_scores = [get_route_score(graph, route) for route in prev_routes] + [0]


    next_availility = compute_availability_matrix(graph, status, next_cards)
    next_routes = get_possible_routes(next_availility)
    next_scores = [get_route_score(graph, route) for route in next_routes] + [0]

    return max(next_scores) - max(prev_scores)


def compute_card_agent_input(graph, status, public_cards, player_cards):
    availability = compute_availability_matrix(graph, status, player_cards)
    array = np.concatenate([graph.flatten(), availability.flatten(), public_cards, player_cards])
    return torch.from_numpy(array).float()
