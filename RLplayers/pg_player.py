import numpy as np
import torch
import torch.nn as nn
from utils.game_utils import compute_availability_matrix, get_available_routes
import torch.nn.functional as F
from torch.distributions import Categorical

def compute_input_matrix(game, players, hide_cards=False):
    dqn_input = None
    for player in players:
        availability = compute_availability_matrix(game.graph, game.status, player)
        availability = torch.from_numpy(np.reshape(availability, (7*7*7, 1))).float()
        cards = torch.zeros((84, 1))
        for i in range(len(player.cards)):
            for j in range(12*i, 12*i + int(player.cards[i])):
                cards[j] = 1
        trains = torch.zeros((10, 1))
        for i in range(int(player.trains)):
            trains[i] = 1
        if dqn_input is None:
            dqn_input = (torch.cat([availability, cards, trains])).T
        else:
            if hide_cards:
                # print("here")
                dqn_input = (torch.cat([dqn_input.T, torch.zeros(availability.shape), torch.zeros(cards.shape), trains])).T
            else:
                dqn_input = (torch.cat([dqn_input.T, availability, cards, trains])).T
    return dqn_input





class PGPlayer:
    def __init__(self, num_colors, destination_cards, trains, id, model):
        self.cards = num_colors * [0]
        self.routes = {}
        self.trains = trains
        self.destination_cards = destination_cards
        self.id = id
        self.net = model
     

    def choose_route(self, game, players, hide_cards=False):
        """
        chooses a route based on the route policy network (route_net)
        and the availability of routes
        """
        availability = compute_availability_matrix(game.graph, game.status, self)
        availability = torch.from_numpy(np.reshape(availability, (7*7*7, 1))).float()
        mask = torch.cat([torch.tensor([[0]]), torch.reshape(availability, (7*7*7, 1))])
        dist = self.net(compute_input_matrix(game, players, hide_cards), mask)
        action = dist.sample()
        action_ = action.item() - 1
        u = action_ // 49
        v = (action_ - 49 * u) // 7
        c = action_ - 49 * u - 7 * v 
        return (u,v,c)

    def draw_or_claim(self, game, players, hide_cards=False):
        """
        Choose the action associated withe the highest value
        based on the card policy
        """
        availability = compute_availability_matrix(game.graph, game.status, self)
        # availability = torch.from_numpy(np.reshape(availability, (7*7*7, 1))).float()
        mask = torch.cat([torch.tensor([[1]]), torch.from_numpy(np.reshape(availability, (7*7*7, 1)))])
        dist = self.net(compute_input_matrix(game, players, hide_cards), mask)
        action = dist.sample()
        if action == 0:
            return 0
        else:
            return 1

    def duplicate(self):
        copy = PGPlayer(len(self.cards), self.destination_cards[:], self.trains, self.id, self.net)
        copy.cards = self.cards[:]
        return copy



class PGNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim = 64, layer = 2):
        """
        Initialize the parameter for the policy network
        """
        super(PGNetwork, self).__init__()
        # Note: here we are given the in_dim and out_dim to the network
        #### Your code here
        if layer == 0:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            modules = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            for i in range(layer-1):
                modules.append(nn.Linear(hidden_dim, hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*modules)
    

    def forward(self, observation, mask):
        """
        This function takes in a batch of observations and a batch of actions, and 
        computes a probability distribution (Categorical) over all (discrete) actions
        
        observation: shape (batch_size, observation_size) torch Tensor
        
        return: a categorical distribution over all possible actions. You may find torch.distributions.Categorical useful
        """
        #### Your code here
        x = self.net(observation)
        logits = F.log_softmax(x, dim=1)
        mask_value = torch.finfo(logits.dtype).min
        # print(mask)
        # print((mask + 1) % 2)
        # print(logits.shape, mask.shape)
        adjusted_logits = torch.where(mask.T > 0, 
                              logits, torch.tensor(mask_value))
        # logits = torch.clone(logits).masked_fill_((mask + 1) % 2, mask_value)
        return Categorical(logits=adjusted_logits)