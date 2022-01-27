import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):

    def __init__(self, input_dims, num_outputs, hidden_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dims[0], hidden_size, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(hidden_size)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        d, w, h = input_dims
        convw = conv2d_size_out(w)
        convh = conv2d_size_out(h)
        linear_input_size = convw * convh  *  32
        self.head = nn.Linear(linear_input_size, num_outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return F.softmax(self.head(x.view(x.size(0), -1)), dim=1)
