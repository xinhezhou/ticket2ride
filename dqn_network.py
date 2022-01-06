import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, device, learning_rate=1e-4):
        super().__init__()
        self.num_actions = num_actions
        self.device = device
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)

        
    def forward(self, x):
        # print(x)
        x = x.to(self.device)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=0)
        return x.view(x.size(0), -1)