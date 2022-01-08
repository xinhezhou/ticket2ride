from game_utils import get_possible_routes, get_route_score, compute_availability_matrix
import numpy as np
import torch


def compute_dqn_input(graph, status, cards):
    array = np.concatenate([graph.flatten(), status.flatten(), cards])
    return torch.from_numpy(array).float()
