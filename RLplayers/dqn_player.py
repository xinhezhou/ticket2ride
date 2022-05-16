import numpy as np
import torch
import torch.nn as nn
from utils.game_utils import compute_availability_matrix, get_available_routes

def compute_input_matrix(game, players):
    player = players[0]
    availability = compute_availability_matrix(game.graph, game.status, player)
    availability = torch.from_numpy(np.reshape(availability, (7*7*7, 1))).float()
    cards = torch.zeros((84, 1))
    for i in range(len(player.cards)):
        for j in range(12*i, 12*i + int(player.cards[i])):
            cards[j] = 1
    trains = torch.zeros((15, 1))
    for i in range(int(player.trains)):
        trains[i] = 1
    
    dqn_input = (torch.cat([availability, cards, trains])).T
    return dqn_input

def compute_output_matrix(game, player):
    availability = compute_availability_matrix(game.graph, game.status, player)
    availability = torch.from_numpy(np.reshape(availability, (7*7*7, 1))).float()

    
    dqn_input = compute_input_matrix(game, [player])
    dqn_output = player.net(dqn_input)
    dqn_output -= torch.min(dqn_output)
    action_dist = torch.mul(torch.transpose(dqn_output, 0, 1)[1:], availability)
    return dqn_output[0][0], action_dist




class DQNPlayer:
    def __init__(self, num_colors, destination_cards, trains, id, model):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains = trains
        self.destination_cards = destination_cards
        self.id = id
        self.net = model
     

    def choose_route(self, game, players):
        """
        chooses a route based on the route policy network (route_net)
        and the availability of routes
        """
        draw_value, action_dist = compute_output_matrix(game, self)
        action = torch.argmax(action_dist).item()
        u = action // 49
        v = (action - 49 * u) // 7
        c = action - 49 * u - 7 * v  
        if u == 0 and v == 0:
            print(draw_value, max(action_dist))
            # print(get_available_routes(compute_availability_matrix(game.graph, game.status, self)))
        return u, v, c
        

    def draw_or_claim(self, game, players, eps=False):
        """
        Choose the action associated withe the highest value
        based on the card policy
        """
        
        draw_value, action_dist = compute_output_matrix(game, self)
        if draw_value > torch.max(action_dist):
            return 0
        else:
            return 1

    def duplicate(self):
        copy = DQNPlayer(len(self.cards), self.destination_cards[:], self.trains, self.id, self.net)
        copy.cards = self.cards[:]
        return copy

   

class QValueNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        """
        Initialize the parameter for the value function
        """
        super(QValueNetwork, self).__init__()
        #### Your code here
        self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )
        

    def forward(self, observation):
        """
        This function takes in a batch of observations, and 
        computes the corresponding batch of values V(s)
        
        observation: shape (batch_size, observation_size) torch Tensor
        
        return: shape (batch_size,) values, i.e. V(observation)
        """
        #### Your code here
        return self.net(observation)