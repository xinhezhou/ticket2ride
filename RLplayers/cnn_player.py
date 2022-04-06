import numpy as np
import torch
from RLplayers.rl_utils import generate_state_matrix
from utils.game_utils import compute_availability_matrix, get_available_routes


class CNNPlayer:
    def __init__(self, num_colors, destination_cards, trains, id, model):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains = trains
        self.destination_cards = destination_cards
        self.id = id
        self.net = model
     

    def choose_route(self, game, players, eps=False):
        """
        chooses a route based on the route policy network (route_net)
        and the availability of routes
        """

        dqn_input = generate_state_matrix(game, players).unsqueeze(0)
        dqn_output = self.net(dqn_input)
        dqn_output -= torch.min(dqn_output)
        availability = compute_availability_matrix(game.graph, game.status, self)
        if eps:
            routes = get_available_routes(availability)
            return routes[np.random.randint(len(routes))]

        availability = np.reshape(availability, (7*7*7, 1))
        
        # print(dqn_output.T.shape)
        action_dist = torch.mul(torch.transpose(dqn_output, 0, 1)[1:], torch.from_numpy(availability))
        action = torch.argmax(action_dist).item()
        u = action // 49
        v = (action - 49 * u) // 7
        c = action - 49 * u - 7 * v  
        if u == 0 and v == 0:
            print(dqn_output[0][0], max(action_dist))
            # print(get_available_routes(compute_availability_matrix(game.graph, game.status, self)))
        return u, v, c
        

    def draw_or_claim(self, game, players, eps=False):
        """
        Choose the action associated withe the highest value
        based on the card policy
        """
        if eps:
            return np.random.randint(2)
        availability = compute_availability_matrix(game.graph, game.status, self)
        availability = np.reshape(availability, (7*7*7, 1))
        dqn_input = generate_state_matrix(game, players).unsqueeze(0)
        dqn_output = self.net(dqn_input)
        dqn_output -= torch.min(dqn_output)
        action_dist = torch.mul(torch.transpose(dqn_output, 0, 1)[1:], torch.from_numpy(availability))
        # print(dqn_output)
        if dqn_output[0][0] > torch.max(action_dist):
            return 0
        else:
            return 1

    def duplicate(self):
        copy = CNNPlayer(len(self.cards), self.destination_cards[:], self.trains, self.id, self.net)
        copy.cards = self.cards[:]
        return copy

   